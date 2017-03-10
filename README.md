# as-auto-sklearn
This project incorporates the auto-sklearn toolkit into a solver runtime 
prediction framework. The predictions directly yield a solution to the algorithm
selection problem.

## Installation

```
cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

**N.B.** If installing under anaconda, use `pip` rather than `pip3`

## Example usage

### Training

```
train-as-auto-sklearn /path/to/coseal/BNSL-2016/ 'model.${solver}.${fold}.jpkl' --total-training-time 120 --iteration-time-limit 30 --config /path/to/my/config.yaml
```

Please try `train-as-auto-sklearn -h` for more options.

### Testing

```
test-as-auto-sklearn /path/to/coseal/BNSL-2016/ 'model.${solver}.${fold}.jpkl bnsl-2016.predictions.csv.gz --config /path/to/my/config.yaml
```

Please try `test-as-auto-sklearn -h` for more options.

---

More documentation is coming soon.