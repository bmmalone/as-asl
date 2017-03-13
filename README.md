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
train-as-auto-sklearn /path/to/coseal/BNSL-2016/ 'model.${solver}.${fold}.gz' --total-training-time 120 --iteration-time-limit 30 --config /path/to/my/config.yaml
```

Please try `train-as-auto-sklearn -h` for more options.

### Testing

```
test-as-auto-sklearn /path/to/coseal/BNSL-2016/ 'model.${solver}.${fold}.gz bnsl-2016.predictions.csv.gz --config /path/to/my/config.yaml
```

Please try `test-as-auto-sklearn -h` for more options.

### Algorithm selection performance

```
validate-as-auto-sklearn /path/to/coseal/BNSL-2016/ bnsl-2016.predictions.csv.gz --config /path/to/my/config.yaml
```

Please try `validate-as-auto-sklearn -h` for more options.

---

Please see the [usage instructions](docs/usage-instructions.md) and [configuration options](docs/config-options.md) for more detailed explanations of running the software.