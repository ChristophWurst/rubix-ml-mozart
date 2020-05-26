<?php

namespace Test\Vendor\Rubix\ML\Regressors;

use Test\Vendor\Rubix\ML\Online;
use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Other\Helpers\Stats;
use Test\Vendor\Rubix\ML\Kernels\Distance\Distance;
use Test\Vendor\Rubix\ML\Kernels\Distance\Euclidean;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function array_slice;

/**
 * KNN Regressor
 *
 * A version of the K Nearest Neighbors algorithm that uses the average (mean) outcome of
 * the *k* nearest data points to an unknown sample in order to make continuous-valued
 * predictions suitable for regression problems.
 *
 * > **Note:** This learner is considered a *lazy* learner because it does the majority
 * of its computation during inference. For a fast spatial tree-accelerated version, see
 * KD Neighbors Regressor.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNRegressor implements Estimator, Learner, Online, Persistable
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
     * The distance kernel to use when computing the distances.
     *
     * @var \Test\Vendor\Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The training samples.
     *
     * @var array[]
     */
    protected $samples = [
        //
    ];

    /**
     * The training labels.
     *
     * @var (string|int|float)[]
     */
    protected $labels = [
        //
    ];

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Test\Vendor\Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = true, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new Euclidean();
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
        return $this->kernel->compatibility();
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
            'kernel' => $this->kernel,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->samples and $this->labels;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->samples = $this->labels = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (float|int)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            [$labels, $distances] = $this->nearest($sample);

            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $outcome = Stats::weightedMean(array_values($labels), $weights);
            } else {
                $outcome = Stats::mean($labels);
            }

            $predictions[] = $outcome;
        }

        return $predictions;
    }

    /**
     * Find the K nearest neighbors to the given sample vector using
     * the brute force method.
     *
     * @param (string|int|float)[] $sample
     * @return array[]
     */
    protected function nearest(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        $distances = array_slice($distances, 0, $this->k, true);

        $labels = array_intersect_key($this->labels, $distances);

        return [$labels, $distances];
    }
}
