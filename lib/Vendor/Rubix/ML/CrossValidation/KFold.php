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
 * K Fold
 *
 * K Fold is a cross validation technique that splits the training set into *k* individual
 * folds and for each training round uses 1 of the folds to test the model and the rest as
 * training data. The final score is the average validation score over all of the *k*
 * rounds. K Fold has the advantage of both training and testing on each sample in the
 * dataset at least once.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KFold implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of folds to split the dataset into.
     *
     * @var int
     */
    protected $k;

    /**
     * @param int $k
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5)
    {
        if ($k < 2) {
            throw new InvalidArgumentException('K must be greater'
                 . " than 1, $k given.");
        }

        $this->k = $k;
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

        $dataset->randomize();

        $folds = $dataset->labelType()->isCategorical()
            ? $dataset->stratifiedFold($this->k)
            : $dataset->fold($this->k);

        $this->backend->flush();

        for ($i = 0; $i < $this->k; ++$i) {
            $training = Labeled::quick();
    
            foreach ($folds as $j => $fold) {
                if ($i !== $j) {
                    $training = $training->merge($fold);
                }
            }

            $testing = $folds[$i];
            
            $this->backend->enqueue(
                new TrainAndValidate($estimator, $training, $testing, $metric)
            );
        }

        $scores = $this->backend->process();

        return Stats::mean($scores);
    }
}
