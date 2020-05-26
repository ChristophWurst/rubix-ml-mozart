<?php

namespace Test\Vendor\Rubix\ML\Classifiers;

use Test\Vendor\Rubix\ML\Online;
use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Verbose;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\Probabilistic;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\NeuralNet\Snapshot;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use Test\Vendor\Rubix\ML\NeuralNet\FeedForward;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Dense;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden;
use Test\Vendor\Rubix\ML\Other\Traits\LoggerAware;
use Test\Vendor\Rubix\ML\Other\Traits\ProbaSingle;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Adam;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Multiclass;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\FBeta;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\Initializers\Xavier1;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function is_nan;
use function count;

/**
 * Multilayer Perceptron
 *
 * A multiclass feed forward neural network classifier with user-defined hidden layers. The
 * Multilayer Perceptron is a deep learning model capable of forming higher-order feature
 * representations through layers of computation. In addition, the MLP features progress
 * monitoring which stops training when it can no longer make progress. It utilizes network
 * snapshotting to make sure that it always has the best model parameters even if progress
 * declined during training.
 *
 * References:
 * [1] G. E. Hinton. (1989). Connectionist learning procedures.
 * [2] L. Prechelt. (1997). Early Stopping - but when?
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MultilayerPerceptron implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use PredictsSingle, ProbaSingle, LoggerAware;

    /**
     * An array composing the user-specified hidden layers of the network in order.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $hiddenLayers;

    /**
     * The number of training samples to process at a time.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to update the network parameters.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before terminating.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The number of epochs without improvement in the validation score to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The proportion of training samples to use for validation and progress
     * monitoring.
     *
     * @var float
     */
    protected $holdOut;

    /**
     * The function that computes the loss associated with an erroneous
     * activation during training.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss
     */
    protected $costFn;

    /**
     * The validation metric used to score the generalization performance of
     * the model during training.
     *
     * @var \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected $metric;

    /**
     * The underlying neural network instance.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\FeedForward|null
     */
    protected $network;

    /**
     * The unique class labels.
     *
     * @var string[]|null
     */
    protected $classes;

    /**
     * The validation scores at each epoch.
     *
     * @var float[]
     */
    protected $scores = [
        //
    ];

    /**
     * The average training loss at each epoch.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[] $hiddenLayers
     * @param int $batchSize
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer|null $optimizer
     * @param float $alpha
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss|null $costFn
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param float $holdOut
     * @throws \InvalidArgumentException
     */
    public function __construct(
        array $hiddenLayers = [],
        int $batchSize = 128,
        ?Optimizer $optimizer = null,
        float $alpha = 1e-4,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 3,
        float $holdOut = 0.1,
        ?ClassificationLoss $costFn = null,
        ?Metric $metric = null
    ) {
        foreach ($hiddenLayers as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Hidden layer'
                    . ' must implement the Hidden interface.');
            }
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut <= 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::check($this, $metric);
        }

        $this->hiddenLayers = $hiddenLayers;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer ?? new Adam();
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->metric = $metric ?? new FBeta();
    }

    /**
     * Return the estimator type.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'hidden_layers' => $this->hiddenLayers,
            'batch_size' => $this->batchSize,
            'optimizer' => $this->optimizer,
            'alpha' => $this->alpha,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
            'window' => $this->window,
            'hold_out' => $this->holdOut,
            'cost_fn' => $this->costFn,
            'metric' => $this->metric,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->network and $this->classes;
    }

    /**
     * Return the validation score at each epoch.
     *
     * @return float[]
     */
    public function scores() : array
    {
        return $this->scores;
    }

    /**
     * Return the training loss at each epoch.
     *
     * @return float[]
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return \Test\Vendor\Rubix\ML\NeuralNet\FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $classes = $dataset->possibleOutcomes();

        $hiddenLayers = $this->hiddenLayers;

        $hiddenLayers[] = new Dense(count($classes), $this->alpha, true, new Xavier1());

        $this->network = new FeedForward(
            new Placeholder1D($dataset->numColumns()),
            $hiddenLayers,
            new Multiclass($classes, $this->costFn),
            $this->optimizer
        );

        $this->classes = $classes;

        $this->scores = $this->steps = [];

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->network) {
            $this->train($dataset);
            
            return;
        }

        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify($this->params()));

            $this->logger->info('Training started');
        }
        
        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$min, $max] = $this->metric->range();

        $bestScore = $min;
        $bestEpoch = $delta = 0;
        $snapshot = null;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $training->randomize()->batch($this->batchSize);

            $loss = 0.0;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            $loss /= count($batches);

            if (is_nan($loss)) {
                throw new RuntimeException('Numerical under/overflow detected.');
            }

            $predictions = $this->predict($testing);

            $score = $this->metric->score($predictions, $testing->labels());

            $this->steps[] = $loss;
            $this->scores[] = $score;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch score=$score loss=$loss");
            }

            if ($score > $bestScore) {
                $bestScore = $score;
                $bestEpoch = $epoch;

                $snapshot = Snapshot::take($this->network);

                $delta = 0;
            } else {
                ++$delta;
            }

            if ($loss <= 0.0 or $score >= $max) {
                break 1;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break 1;
            }

            if ($delta >= $this->window) {
                break 1;
            }

            $prevLoss = $loss;
        }

        if (end($this->scores) < $bestScore) {
            if ($snapshot) {
                $snapshot->restore();
                
                if ($this->logger) {
                    $this->logger->info('Parameters restored from'
                        . " snapshot at epoch $bestEpoch.");
                }
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Test\Vendor\Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->network or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $activations = $this->network->infer($dataset);

        $probabilities = [];

        foreach ($activations->asArray() as $dist) {
            $probabilities[] = array_combine($this->classes, $dist) ?: [];
        }

        return $probabilities;
    }
}
