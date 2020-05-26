<?php

namespace Test\Vendor\Rubix\ML\Classifiers;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Parallel;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\Probabilistic;
use Test\Vendor\Rubix\ML\RanksFeatures;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Backends\Serial;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Backends\Tasks\Proba;
use Test\Vendor\Rubix\ML\Backends\Tasks\Predict;
use Test\Vendor\Rubix\ML\Other\Traits\ProbaSingle;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Backends\Tasks\TrainLearner;
use Test\Vendor\Rubix\ML\Other\Traits\Multiprocessing;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Test\Vendor\Rubix\ML\argmax;
use function Test\Vendor\Rubix\ML\array_transpose;
use function get_class;
use function in_array;

/**
 * Random Forest
 *
 * An ensemble classifier that trains Decision Trees (Classification or Extra Trees) on random
 * subsets (*bootstrap* set) of the training data. Predictions are based on the probability
 * scores returned from each tree in the forest, averaged and weighted equally.
 *
 * References:
 * [1] L. Breiman. (2001). Random Forests.
 * [2] L. Breiman et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomForest implements Estimator, Learner, Probabilistic, Parallel, RanksFeatures, Persistable
{
    use Multiprocessing, PredictsSingle, ProbaSingle;

    /**
     * The class names of the learners that the ensemble is compatible with.
     *
     * @var string[]
     */
    public const COMPATIBLE_LEARNERS = [
        ClassificationTree::class,
        ExtraTreeClassifier::class,
    ];

    /**
     * The base learner.
     *
     * @var \Test\Vendor\Rubix\ML\Learner
     */
    protected $base;

    /**
     * The number of trees to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The ratio of training samples to train each decision tree on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * Should we sample the bootstrap set to compensate for imbalanced class labels?
     *
     * @var bool
     */
    protected $balanced;

    /**
     * The decision trees that make up the forest.
     *
     * @var mixed[]|null
     */
    protected $trees;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]|null
     */
    protected $classes;

    /**
     * The number of feature columns in the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param \Test\Vendor\Rubix\ML\Learner|null $base
     * @param int $estimators
     * @param float $ratio
     * @param bool $balanced
     * @throws \InvalidArgumentException
     */
    public function __construct(
        ?Learner $base = null,
        int $estimators = 100,
        float $ratio = 0.2,
        bool $balanced = false
    ) {
        if ($base and !in_array(get_class($base), self::COMPATIBLE_LEARNERS)) {
            throw new InvalidArgumentException('Base Learner must be'
                . ' compatible with ensemble.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.5) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1.5, $ratio given.");
        }

        $this->base = $base ?? new ClassificationTree();
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->balanced = $balanced;
        $this->backend = new Serial();
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
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'base' => $this->base,
            'estimators' => $this->estimators,
            'ratio' => $this->ratio,
            'balanced' => $this->balanced,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->trees);
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
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $k = (int) ceil($this->ratio * $dataset->numRows());

        if ($this->balanced) {
            $counts = array_count_values($dataset->labels());

            $min = min($counts);

            $weights = [];

            foreach ($dataset->labels() as $label) {
                $weights[] = $min / $counts[$label];
            }
        }

        $this->backend->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $estimator = clone $this->base;

            if (isset($weights)) {
                $subset = $dataset->randomWeightedSubsetWithReplacement($k, $weights);
            } else {
                $subset = $dataset->randomSubsetWithReplacement($k);
            }

            $this->backend->enqueue(new TrainLearner($estimator, $subset));
        }

        $this->trees = $this->backend->process();

        $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);

        $this->featureCount = $dataset->numColumns();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trees) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->trees as $estimator) {
            $this->backend->enqueue(new Predict($estimator, $dataset));
        }

        $aggregate = array_transpose($this->backend->process());

        $predictions = [];

        foreach ($aggregate as $votes) {
            $predictions[] = argmax(array_count_values($votes));
        }

        return $predictions;
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
        if (!$this->trees or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = array_fill(0, $dataset->numRows(), $this->classes);

        $this->backend->flush();

        foreach ($this->trees as $estimator) {
            $this->backend->enqueue(new Proba($estimator, $dataset));
        }

        $aggregate = $this->backend->process();

        foreach ($aggregate as $proba) {
            foreach ($proba as $i => $joint) {
                foreach ($joint as $class => $probability) {
                    $probabilities[$i][$class] += $probability;
                }
            }
        }

        foreach ($probabilities as &$joint) {
            foreach ($joint as &$probability) {
                $probability /= $this->estimators;
            }
        }

        return $probabilities;
    }

    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!$this->trees or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this->trees as $tree) {
            foreach ($tree->featureImportances() as $column => $importance) {
                $importances[$column] += $importance;
            }
        }

        foreach ($importances as &$importance) {
            $importance /= $this->estimators;
        }

        return $importances;
    }
}
