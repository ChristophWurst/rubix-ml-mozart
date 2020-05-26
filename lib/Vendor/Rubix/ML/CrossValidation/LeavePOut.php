<?php

namespace Test\Vendor\Rubix\ML\CrossValidation;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Parallel;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Backends\Serial;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Helpers\Stats;
use Test\Vendor\Rubix\ML\Other\Traits\Multiprocessing;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric;
use Test\Vendor\Rubix\ML\Backends\Tasks\TrainAndValidate;
use Test\Vendor\Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use InvalidArgumentException;

/**
 * Leave P Out
 *
 * Leave P Out tests a learner with a unique holdout set of size p for each iteration until
 * all samples have been tested. Although Leave P Out can take long with large datasets and
 * small values of p, it is especially suited for small datasets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeavePOut implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of samples to leave out each round for testing.
     *
     * @var int
     */
    protected $p;

    /**
     * @param int $p
     * @throws \InvalidArgumentException
     */
    public function __construct(int $p = 10)
    {
        if ($p < 1) {
            throw new InvalidArgumentException('P must be greater'
                . " than 0, $p given.");
        }

        $this->p = $p;
        $this->backend = new Serial();
    }

    /**
     * Test the estimator with the supplied dataset and return a validation score.
     *
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \InvalidArgumentException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::check($estimator, $metric);

        $n = (int) round($dataset->numRows() / $this->p);

        $this->backend->flush();

        for ($i = 0; $i < $n; ++$i) {
            $training = clone $dataset;

            $testing = $training->splice($i * $this->p, $this->p);

            $this->backend->enqueue(
                new TrainAndValidate($estimator, $training, $testing, $metric)
            );
        }

        $scores = $this->backend->process();

        return Stats::mean($scores);
    }
}
