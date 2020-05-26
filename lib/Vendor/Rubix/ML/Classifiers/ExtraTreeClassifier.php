<?php

namespace Test\Vendor\Rubix\ML\Classifiers;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\Probabilistic;
use Test\Vendor\Rubix\ML\RanksFeatures;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Graph\Nodes\Best;
use Test\Vendor\Rubix\ML\Graph\Nodes\Outcome;
use Test\Vendor\Rubix\ML\Graph\Trees\ExtraTree;
use Test\Vendor\Rubix\ML\Other\Traits\ProbaSingle;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Extra Tree Classifier
 *
 * An Extremely Randomized Classification Tree that recursively chooses node splits
 * with the least entropy among a set of *k* (given by max features) completely
 * random split points. Extra Trees are useful in ensembles such as Random Forest or
 * AdaBoost as the *weak* learner or they can be used on their own. The strength of
 * Extra Trees as compared to more greedy decision trees are their computational
 * efficiency and reduced bias.
 *
 * References:
 * [1] P. Geurts et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ExtraTreeClassifier extends ExtraTree implements Estimator, Learner, Probabilistic, RanksFeatures, Persistable
{
    use PredictsSingle, ProbaSingle;
    
    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]|null
     */
    protected $classes;

    /**
     * @param int $maxDepth
     * @param int $maxLeafSize
     * @param int|null $maxFeatures
     * @param float $minPurityIncrease
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxDepth = PHP_INT_MAX,
        int $maxLeafSize = 3,
        ?int $maxFeatures = null,
        float $minPurityIncrease = 1e-7
    ) {
        parent::__construct($maxDepth, $maxLeafSize, $maxFeatures, $minPurityIncrease);
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
            DataType::categorical(),
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
            'max_depth' => $this->maxDepth,
            'max_leaf_size' => $this->maxLeafSize,
            'max_features' => $this->maxFeatures,
            'min_purity_increase' => $this->minPurityIncrease,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare();
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

        $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);

        $this->grow($dataset);
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
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Best
                ? $node->outcome()
                : '?';
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
        if ($this->bare() or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset->samples() as $sample) {
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Best
                ? array_replace($this->classes, $node->probabilities()) ?? []
                : [];
        }

        return $probabilities;
    }

    /**
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @return \Test\Vendor\Rubix\ML\Graph\Nodes\Outcome
     */
    protected function terminate(Labeled $dataset) : Outcome
    {
        $n = $dataset->numRows();

        $counts = array_count_values($dataset->labels());

        $max = max($counts);

        $outcome = (string) array_search($max, $counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $p = $n ? $max / $n : 1.0;

        $impurity = -($p * log($p));

        return new Best($outcome, $probabilities, $impurity, $n);
    }

    /**
     * Compute the entropy of a labeled dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    protected function impurity(Labeled $dataset) : float
    {
        $n = $dataset->numRows();

        if ($n <= 1) {
            return 0.0;
        }

        $counts = array_count_values($dataset->labels());

        $entropy = 0.0;

        foreach ($counts as $count) {
            $p = $count / $n;

            $entropy -= $p * log($p);
        }

        return $entropy;
    }
}
