from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
        import optuna

import json
import os
import time
import uuid
from pathlib import Path

import optuna


def _atomic_move(src: Path, dst: Path) -> None:
    # Atomic move using os.replace (assumes same filesystem).
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _atomic_write_json(dst: Path, payload: dict) -> None:
    # Atomic JSON write using temp file and replace.
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".tmp.{dst.name}.{uuid.uuid4().hex}")
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def _normalize_values(values: Any) -> Any:
    """
    Normalize objective values for Optuna:
      - single objective: float
      - multi objective: list[float]
      - None is allowed only for FAIL states (handled by caller)
    """
    if values is None:
        return None
    if isinstance(values, (int, float)):
        return float(values)
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for v in values:
            if not isinstance(v, (int, float)):
                raise ValueError(f"values contains non-numeric item: {v!r}")
            out.append(float(v))
        return out
    raise ValueError(f"unsupported values type: {type(values).__name__}")


def commit_finished_results_to_optuna_dbs(
    finished_dir: str,
    *,
    db_dir: str = ".",
    committed_dir: str = "optuna_committed_results",
) -> None:
    """
    Read finished result JSON files from finished_dir and commit them to
    per-study Optuna SQLite DB files named f"{study_name}.db".

    After successful commit, move the JSON file to committed_dir.

    - finished_dir: pass args.optuna_finished_predefined_trials_dir directly.
    - If the target study does not exist, raise an error (do not create).
    - Errors are not swallowed; this function raises exceptions.
    """
    finished_path = Path(finished_dir)
    db_base = Path(db_dir)
    committed_path = Path(committed_dir)

    if not finished_path.exists():
        raise FileNotFoundError(f"finished_dir does not exist: {finished_path}")

    for fp in sorted(finished_path.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{fp}: finished result must be a JSON object")

        # Mandatory fields.
        study_name = data.get("study_name")
        trial_number = data.get("trial_number")
        state_str = data.get("state")
        values = data.get("values", None)

        if not isinstance(study_name, str) or not study_name:
            raise ValueError(f"{fp}: missing/invalid study_name")
        if not isinstance(trial_number, int):
            raise ValueError(f"{fp}: missing/invalid trial_number")
        if not isinstance(state_str, str) or not state_str:
            raise ValueError(f"{fp}: missing/invalid state")

        db_path = db_base / f"{study_name}.db"
        storage_url = f"sqlite:///{db_path.as_posix()}"

        # Do NOT auto-create. Study must already exist.
        study = optuna.load_study(study_name=study_name, storage=storage_url)

        if state_str == "COMPLETE":
            norm_values = _normalize_values(values)
            study.tell(trial_number, norm_values)
        elif state_str == "FAIL":
            study.tell(trial_number, state=TrialState.FAIL)
        else:
            raise ValueError(f"{fp}: unsupported state={state_str!r}")

        # After successful commit, move the JSON to committed_dir.
        committed_dst = committed_path / fp.name
        _atomic_move(fp, committed_dst)


def run_predefined_trial(
    path_str: str,
    objective,
    *,
    invalid_dir: str = "optuna_invalid_predefined_trials",
    running_dir: str = "optuna_running_predefined_trials",
    finished_dir: str = "optuna_finished_predefined_trials",
) -> None:
    """
    Execute a predefined trial file.

    The input argument is a string path, typically provided directly from
    command line arguments.

    Workflow:
      1) Atomically move the input file to the running directory.
      2) Load and validate it as a PredefinedTrial.
      3) If validation fails, move it to the invalid directory and stop.
      4) If validation succeeds, run objective(trial).
      5) Write results to the finished directory.
      6) Remove the running file.
    """

    from pathlib import Path

    path = Path(path_str)

    invalid_dir = Path(invalid_dir)
    running_dir = Path(running_dir)
    finished_dir = Path(finished_dir)

    # ---------------------------------------------------------------
    # 1) Move the original file to the running directory (atomic).
    # ---------------------------------------------------------------
    unique_name = f"{path.stem}.running.{int(time.time())}.{uuid.uuid4().hex}{path.suffix}"
    running_path = running_dir / unique_name

    try:
        _atomic_move(path, running_path)
    except Exception as e:
        invalid_dir.mkdir(parents=True, exist_ok=True)
        invalid_path = invalid_dir / f"{path.stem}.move_error.{int(time.time())}.{uuid.uuid4().hex}{path.suffix}"
        _atomic_write_json(
            invalid_path,
            {"error": {"type": type(e).__name__, "message": str(e)}},
        )
        return

    # ---------------------------------------------------------------
    # 2) Load and validate the moved file.
    # ---------------------------------------------------------------
    try:
        trial = load_predefined_trial(running_path)

        with running_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        study_name = data.get("study_name") 
        trial_number = data.get("trial_number") if isinstance(data, dict) else None

        if not isinstance(study_name, str) or not study_name:
            raise ValueError("Missing or invalid study_name in predefined trial file.")

        if not isinstance(trial_number, int):
            raise ValueError("Missing or invalid trial_number in predefined trial file.")

    except Exception:
        invalid_dir.mkdir(parents=True, exist_ok=True)
        invalid_name = f"{running_path.stem}.invalid.{int(time.time())}.{uuid.uuid4().hex}{running_path.suffix}"
        invalid_path = invalid_dir / invalid_name

        try:
            _atomic_move(running_path, invalid_path)
        except Exception:
            try:
                running_path.unlink()
            except FileNotFoundError:
                pass
        return

    # ---------------------------------------------------------------
    # 3) Execute objective(trial).
    # ---------------------------------------------------------------
    started_unix = time.time()

    try:
        values = objective(trial)
        state = "COMPLETE"
        error = None
    except Exception as e:
        values = None
        state = "FAIL"
        error = {"type": type(e).__name__, "message": str(e)}

    finished_unix = time.time()

    trial_number = data.get("trial_number") if isinstance(data, dict) else None

    # ---------------------------------------------------------------
    # 4) Write results to finished directory.
    # ---------------------------------------------------------------
    finished_dir.mkdir(parents=True, exist_ok=True)
    finished_name = f"{running_path.stem}.finished.{int(time.time())}.{uuid.uuid4().hex}{running_path.suffix}"
    finished_path = finished_dir / finished_name

    _atomic_write_json(
        finished_path,
        {
            "study_name": study_name,
            "trial_number": trial_number,
            "values": values,
            "state": state,
            "timing": {
                "started_unix": started_unix,
                "finished_unix": finished_unix,
            },
            "error": error,
        },
    )

    # ---------------------------------------------------------------
    # 5) Remove running file.
    # ---------------------------------------------------------------
    try:
        running_path.unlink()
    except FileNotFoundError:
        pass


def save_predefined_params_from_trial(
    path: str | Path,
    trial: optuna.trial.Trial,
    study_name: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a predefined-parameter JSON config from an Optuna `Trial` (suggest already executed).

    Assumptions / contract
    ----------------------
    - This function must be called **after** all required `trial.suggest_*()` calls,
      because `trial.params` contains only *suggested so far* parameters.
    - Categorical parameters are expected to be represented as JSON-serializable values
      (recommended: **int IDs**). If `trial.params` contains non-JSON-serializable objects
      (e.g., classes/functions), this function raises TypeError.

    Output format (wrapped)
    -----------------------
    {
      "params": { ... },              # copied from trial.params
      "trial_number": <int>,          # trial.number
      "created_unix": <float>,        # time.time()
      "study_name": <str>             # study_name
      "metadata": { ... }             # optional
    }

    Atomicity
    ---------
    Writes to a temporary file and then atomically renames to `path` to avoid partial reads.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination JSON path (e.g., queue/000042.json).
    trial : optuna.trial.Trial
        Trial object on which `suggest_*()` has already been called.
    study_name: str
        A study_name
    metadata : dict, optional (keyword-only)
        Extra metadata to store (must be JSON-serializable).

    Raises
    ------
    TypeError
        If payload contains non-JSON-serializable values.
    ValueError
        If `trial.params` is empty (likely called before suggestions).
    """

    import optuna

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    params = dict(trial.params)
    if not params:
        raise ValueError(
            "trial.params is empty. Call this function only after executing "
            "all required trial.suggest_*() calls."
        )

    # study_name must be explicitly provided.
    if not isinstance(study_name, str) or not study_name:
        raise ValueError("study_name must be a non-empty string.")

    payload: Dict[str, Any] = {
        "params": params,
        "trial_number": int(trial.number),
        "created_unix": time.time(),
        "study_name": study_name,
    }
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict if provided")
        payload["metadata"] = metadata

    # Fail fast if not JSON-serializable
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    except TypeError as e:
        raise TypeError(f"Non-JSON-serializable value detected in payload: {e}") from e

    # Atomic write: tmp + replace
    tmp_path = path.with_name(f".tmp.{path.name}.{uuid.uuid4().hex}")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_predefined_trial(path):
    """
    Load a parameter JSON file and construct a PredefinedTrial.

    This function reads a JSON file generated by the ask-side coordinator,
    extracts stored parameters, and returns a PredefinedTrial instance.

    Supported JSON formats
    -----------------------
    1) Raw parameter mapping:
       {
           "x0": 1,
           "x1": 3,
           "optimizer": 0
       }

    2) Wrapped format:
       {
           "params": {
               "x0": 1,
               "x1": 3,
               "optimizer": 0
           },
           "trial_number": 42,
           ...
       }

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file containing predefined parameters.

    Returns
    -------
    PredefinedTrial
        Read-only trial-compatible object backed by stored parameters.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the JSON does not contain a valid parameter mapping.
    json.JSONDecodeError
        If the file is not valid JSON.

    Examples
    --------
    >>> import tempfile, json
    >>> from pathlib import Path
    >>> p = Path(tempfile.gettempdir()) / "params.json"
    >>> _ = p.write_text(json.dumps({"x": 2, "y": 0.5}))
    >>> trial = load_predefined_trial(p)
    >>> trial.suggest_int("x", 0, 10)
    2
    >>> trial.suggest_float("y", 0.0, 1.0)
    0.5
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # unwrap params if needed
    if isinstance(data, dict) and "params" in data:
        params = data["params"]
    else:
        params = data

    if not isinstance(params, dict):
        raise ValueError(
            f"Invalid parameter JSON format in {path}: "
            f"expected dict or dict['params'], got {type(params)}"
        )

    # defensive copy & minimal validation
    clean_params = {}
    for k, v in params.items():
        if not isinstance(k, str):
            raise ValueError(f"Invalid parameter name (not str): {k!r}")
        clean_params[k] = v

    return PredefinedTrial(clean_params)


class PredefinedTrial:
    """
    Read-only, Trial-compatible object backed by pre-defined parameters.

    This class is designed to be used in distributed optimization setups
    where parameter suggestion (ask) and objective evaluation are separated.

    Unlike optuna.trial.Trial, this object:
      - Does NOT sample parameters
      - Does NOT access Optuna storage
      - Simply returns pre-defined parameter values

    It is intended to be passed into an existing `objective(trial)` function
    without modifying the objective code.

    Key design decisions
    --------------------
    - All parameters must be defined upfront via `params`
    - `suggest_*` methods return stored values only
    - Distribution/range arguments are ignored
    - Missing parameters raise NotImplementedError (fail fast)
    - Categorical parameters are represented as **integer identifiers**

    This makes the class safe for:
      - JSON / filesystem-based parameter passing
      - Multi-process / multi-node execution
      - Reproducible experiments without pickling Python objects

    Examples
    --------
    Basic usage with an Optuna-style objective:

    >>> def objective(trial):
    ...     x = trial.suggest_int("x", 0, 10)
    ...     y = trial.suggest_float("y", 0.0, 1.0)
    ...     z = trial.suggest_categorical("z", ["a", "b", "c"])
    ...     return x + y + z
    ...
    >>> params = {"x": 3, "y": 0.5, "z": 2}
    >>> trial = PredefinedTrial(params)
    >>> objective(trial)
    5.5
    """

    def __init__(self, params: dict):
        """
        Initialize a PredefinedTrial with fixed parameter values.

        Parameters
        ----------
        params : dict
            Mapping from parameter name to value.
            Values must already be validated by the ask/sampling side.

        Notes
        -----
        The dictionary is copied internally to avoid accidental mutation.

        Examples
        --------
        >>> t = PredefinedTrial({"a": 1})
        >>> t.params
        {'a': 1}
        """
        self.params = dict(params)
        self._used = set()

    def _get(self, name: str):
        """
        Internal helper to retrieve a parameter value.

        Raises
        ------
        NotImplementedError
            If the parameter name is not defined.
        """
        if name not in self.params:
            raise NotImplementedError(
                f"Parameter '{name}' is not defined in PredefinedTrial."
            )
        self._used.add(name)
        return self.params[name]

    def suggest_int(self, name, low=None, high=None, step=None, log=False):
        """
        Return a pre-defined integer parameter.

        This method ignores all range-related arguments and simply returns
        the stored value converted to `int`.

        Parameters
        ----------
        name : str
            Parameter name.
        low, high, step, log :
            Ignored. Present only for Optuna API compatibility.

        Returns
        -------
        int

        Examples
        --------
        >>> trial = PredefinedTrial({"n_layers": 4})
        >>> trial.suggest_int("n_layers", 1, 10)
        4
        """
        return int(self._get(name))

    def suggest_float(self, name, low=None, high=None, step=None, log=False):
        """
        Return a pre-defined floating-point parameter.

        This method ignores all range-related arguments and simply returns
        the stored value converted to `float`.

        Parameters
        ----------
        name : str
            Parameter name.
        low, high, step, log :
            Ignored. Present only for Optuna API compatibility.

        Returns
        -------
        float

        Examples
        --------
        >>> trial = PredefinedTrial({"lr": 0.001})
        >>> trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        0.001
        """
        return float(self._get(name))

    def suggest_categorical(self, name, choices):
        """
        Return a pre-defined categorical parameter **as an integer ID**.

        IMPORTANT
        ---------
        This method ALWAYS returns an `int`.

        In this system, categorical parameters are represented as integer
        identifiers (e.g., enum indices or symbolic IDs), not arbitrary
        Python objects.

        The `choices` argument is accepted only for interface compatibility
        with Optuna's Trial API and is intentionally ignored.

        Design rationale
        ----------------
        - Safe JSON / filesystem serialization
        - No reliance on pickle or Python object identity
        - Stable behavior across processes and environments
        - Clear separation between parameter ID and runtime object

        Parameters
        ----------
        name : str
            Parameter name.
        choices : Sequence[Any]
            Ignored. Present only for Optuna API compatibility.

        Returns
        -------
        int
            Integer identifier of the categorical choice.

        Raises
        ------
        NotImplementedError
            If `name` is not defined in this PredefinedTrial instance.

        Examples
        --------
        >>> params = {"optimizer": 0, "activation": 2}
        >>> trial = PredefinedTrial(params)
        >>> trial.suggest_categorical("optimizer", ["adam", "sgd"])
        0
        >>> trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        2

        Typical usage
        -------------
        >>> OPTIMIZERS = {0: "Adam", 1: "SGD"}
        >>> opt_id = trial.suggest_categorical("optimizer", OPTIMIZERS.keys())
        >>> OPTIMIZERS[opt_id]
        'Adam'
        """
        return int(self._get(name))

    # --- Optional Optuna compatibility (no-op) ---

    def report(self, value, step):
        """
        No-op placeholder for Optuna compatibility.

        Pruning is intentionally disabled in PredefinedTrial.
        """
        pass

    def should_prune(self):
        """
        Always return False.

        Pruning is not supported in PredefinedTrial-based evaluation.

        Examples
        --------
        >>> PredefinedTrial({"x": 1}).should_prune()
        False
        """
        return False
