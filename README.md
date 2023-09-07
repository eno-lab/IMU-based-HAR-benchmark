# IMU-based HAR Benchmark 

This is a benchmark utility to benchmark IMU-based HAR models. 

Features
- benchmark setup with iSPLInception's proposal
- Leave one subject out CV for ucihar, daphnet, and pamap2 (new)

The original is the benchmark system publiced on iSPLInception ( https://github.com/rmutegeki/iSPLInception/ ).

# How to adding your models
Please manage your models with a separate repository with add-on style (copy and combined to this benchmark).

An example of separate model repository is 
- README.md
- models/your\_model\_py
- for\_main.patch

It allow you to select license of your model's source code.
