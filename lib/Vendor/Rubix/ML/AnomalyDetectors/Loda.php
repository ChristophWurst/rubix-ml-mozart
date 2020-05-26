<?php

namespace Test\Vendor\Rubix\ML\AnomalyDetectors;

use TEST_Tensor\TEST_Matrix;
use TEST_Tensor\TEST_Vector;
use Test\Vendor\Rubix\ML\Online;
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

use const Test\Vendor\Rubix\ML\LOG_EPSILON;

/**
 * Loda
 *
 * *Lightweight Online Detector of Anomalies* uses a series of sparse random
 * projection vectors to produce scalar inputs to an ensemble of unique
 * one-dimensional equi-width histograms. The histograms are then used to estimate
 * the probability density of an unknown sample during inference.
 *
 * References:
 * [1] T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
 * [2] L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Loda implements Estimator, Learner, Online, Ranking, Persistable
{
    use PredictsSingle, RanksSingle;

    /**
     * The minimum number of histogram bins.
     *
     * @var int
     */
    protected const MIN_BINS = 3;
    
    /**
     * The minimum dimensionality required to produce sparse projections.
     *
     * @var int
     */
    protected const MIN_SPARSE_DIMENSIONS = 3;

    /**
     * The number of projection/histogram pairs in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The number of bins in each equi-width histogram.
     *
     * @var int|null
     */
    protected $bins;

    /**
     * Should we calculate the equi-width bin count on the fly?
     *
     * @var bool
     */
    protected $fitBins;

    /**
     * The proportion of outliers that are assumed to be present in the
     * training set.
     *
     * @var float
     */
    protected $contamination;

    /**
     * The sparse random projection matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $r;

    /**
     * The edges, and bin counts of each histogram.
     *
     * @var array[]
     */
    protected $histograms = [
        //
    ];

    /**
     * The minimum negative log likelihood score necessary to flag an anomaly.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * The number of samples that have been learned so far.
     *
     * @var int
     */
    protected $n = 0;

    /**
     * Estimate the number of bins from the number of samples in a dataset.
     *
     * @param int $n
     * @return int
     */
    public static function estimateBins(int $n) : int
    {
        return (int) round(log($n, 2)) + 1;
    }

    /**
     * @param int $estimators
     * @param int|null $bins
     * @param float $contamination
     * @throws \InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?int $bins = null, float $contamination = 0.1)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if (isset($bins) and $bins < self::MIN_BINS) {
            throw new InvalidArgumentException('Bins must be greater'
                . ' than ' . self::MIN_BINS . ", $bins given.");
        }

        if ($contamination < 0.0 or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->estimators = $estimators;
        $this->bins = $bins;
        $this->fitBins = is_null($bins);
        $this->contamination = $contamination;
    }

    /**
     * Return the integer encoded estimator type.
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
            'estimators' => $this->estimators,
            'bins' => $this->bins,
            'contamination' => $this->contamination,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->r and $this->histograms and $this->threshold;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        [$m, $n] = $dataset->shape();

        if ($this->fitBins) {
            $this->bins = max(self::estimateBins($m), self::MIN_BINS);
        }

        $this->r = TEST_Matrix::gaussian($n, $this->estimators);

        if ($n >= self::MIN_SPARSE_DIMENSIONS) {
            $mask = TEST_Matrix::rand($n, $this->estimators)
                ->less(sqrt($n) / $n);

            $this->r = $this->r->multiply($mask);
        }

        $projections = TEST_Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($projections->asArray() as $values) {
            $edges = TEST_Vector::linspace(min($values), max($values), $this->bins - 1)->asArray();

            $counts = array_fill(0, count($edges), 0);

            foreach ($values as $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }
            }

            $this->histograms[] = [$edges, $counts];
        }

        $this->n = $m;

        $densities = $this->densities($projections);

        $this->threshold = Stats::quantile($densities, 1.0 - $this->contamination);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->r or !$this->histograms or !$this->threshold) {
            $this->train($dataset);

            return;
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $projections = TEST_Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($projections->asArray() as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            foreach ($values as $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }
            }

            $this->histograms[$i] = [$edges, $counts];
        }

        $n = $dataset->numRows();

        $this->n += $n;

        $densities = $this->densities($projections);

        $threshold = Stats::quantile($densities, 1.0 - $this->contamination);

        $weight = $n / $this->n;

        $this->threshold = (1.0 - $weight) * $this->threshold + $weight * $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'decide'], $this->rank($dataset));
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
        if (!$this->r or !$this->histograms or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $projections = TEST_Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        return $this->densities($projections);
    }

    /**
     * Estimate the probability density function of each 1-dimensional projection
     * using the histograms generated during training.
     *
     * @param \TEST_Tensor\TEST_Matrix $projections
     * @return float[]
     */
    protected function densities(TEST_Matrix $projections) : array
    {
        $densities = array_fill(0, $projections->n(), 0.0);
    
        foreach ($projections->asArray() as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            foreach ($values as $j => $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        $count = $counts[$k];

                        $densities[$j] += $count > 0
                            ? -log($count / $this->n)
                            : -LOG_EPSILON;

                        break 1;
                    }
                }
            }
        }

        foreach ($densities as &$density) {
            $density /= $this->estimators;
        }

        return $densities;
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
