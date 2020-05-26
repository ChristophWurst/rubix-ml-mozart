<?php

namespace Test\Vendor\Rubix\ML\Regressors;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Graph\Trees\KDTree;
use Test\Vendor\Rubix\ML\Graph\Trees\Spatial;
use Test\Vendor\Rubix\ML\Other\Helpers\Stats;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors Regressor
 *
 * A fast implementation of KNN Regressor using a spatially-aware binary tree for nearest
 * neighbors search. K-d Neighbors Regressor works by locating the neighborhood of a sample
 * via binary search and then does a brute force search only on the samples close to or
 * within the neighborhood of the unknown sample. The main advantage of K-d Neighbors over
 * brute force KNN is inference speed, however you no longer have the ability to partially
 * train.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNeighborsRegressor implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;
    
    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The spatial tree used to run nearest neighbor searches.
     *
     * @var \Test\Vendor\Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Test\Vendor\Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = true, ?Spatial $tree = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->tree = $tree ?? new KDTree();
    }

    /**
     * Return the estimator type.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->tree->kernel()->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'k' => $this->k,
            'weighted' => $this->weighted,
            'tree' => $this->tree,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->tree->bare();
    }

    /**
     * Return the base k-d tree instance.
     *
     * @var \Test\Vendor\Rubix\ML\Graph\Trees\Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
    }

    /**
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

        $this->tree->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $predictions[] = Stats::weightedMean($labels, $weights);
            } else {
                $predictions[] = Stats::mean($labels);
            }
        }

        return $predictions;
    }
}
