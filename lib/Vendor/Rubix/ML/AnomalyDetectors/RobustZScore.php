<?php

namespace Test\Vendor\Rubix\ML\AnomalyDetectors;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Ranking;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Helpers\Stats;
use Test\Vendor\Rubix\ML\Other\Traits\RanksSingle;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Robust Z-Score
 *
 * A statistical anomaly detector that uses modified Z-Scores which are robust to preexisting
 * outliers in the training set. The modified Z-Score uses the median and median absolute
 * deviation (MAD) unlike the mean and standard deviation of a standard Z-Score - which are
 * more sensitive to outliers. Anomalies are flagged if their final weighted Z-Score exceeds a
 * user-defined threshold.
 *
 * > **Note:** An alpha value of 1 means the estimator only considers the maximum absolute Z-Score,
 * whereas a setting of 0 indicates that only the average Z-Score factors into the final score.
 *
 * References:
 * [1] B. Iglewicz et al. (1993). How to Detect and Handle Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Estimator, Learner, Ranking, Persistable
{
    use PredictsSingle, RanksSingle;
    
    /**
     * The expected value of the MAD as n goes to ∞.
     *
     * @var float
     */
    protected const ETA = 0.6745;

    /**
     * The minimum z score to be flagged as an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The weight of the maximum per sample z score in the overall anomaly score.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The median of each feature column in the training set.
     *
     * @var float[]
     */
    protected $medians = [
        //
    ];

    /**
     * The median absolute deviation of each feature column.
     *
     * @var float[]
     */
    protected $mads = [
        //
    ];

    /**
     * @param float $threshold
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5, float $alpha = 0.5)
    {
        if ($threshold <= 0.0) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        if ($alpha < 0.0 or $alpha > 1.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " between 0 and 1, $alpha given.");
        }

        $this->threshold = $threshold;
        $this->alpha = $alpha;
    }

    /**
     * Return the estimator type.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
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
            'threshold' => $this->threshold,
            'alpha' => $this->alpha,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->medians and $this->mads;
    }

    /**
     * Return the array of computed feature column medians.
     *
     * @return float[]|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the array of computed feature column median absolute deviations.
     *
     * @return float[]|null
     */
    public function mads() : ?array
    {
        return $this->mads;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $this->medians = $this->mads = [];

        foreach ($dataset->columns() as $column => $values) {
            [$median, $mad] = Stats::medianMad($values);

            $this->medians[$column] = $median;
            $this->mads[$column] = $mad ?: EPSILON;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'z'], $dataset->samples());
    }

    /**
     * Calculate the modified z score for a given sample.
     *
     * @param (int|float)[] $sample
     * @return float
     */
    protected function z(array $sample) : float
    {
        $z = [];

        foreach ($sample as $column => $value) {
            $z[] = abs(
                (self::ETA * ($value - $this->medians[$column]))
                / $this->mads[$column]
            );
        }

        return (1.0 - $this->alpha) * Stats::mean($z)
            + $this->alpha * max($z);
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->threshold ? 1 : 0;
    }
}
