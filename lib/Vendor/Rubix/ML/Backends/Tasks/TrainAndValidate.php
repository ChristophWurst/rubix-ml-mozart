<?php

namespace Test\Vendor\Rubix\ML\Backends\Tasks;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric;

class TrainAndValidate extends Task
{
    /**
     * Train the learner and then return its validation score.
     *
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $training
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $testing
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public static function score(
        Learner $estimator,
        Dataset $training,
        Labeled $testing,
        Metric $metric
    ) : float {
        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        return $metric->score($predictions, $testing->labels());
    }

    /**
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $training
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $testing
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric $metric
     */
    public function __construct(
        Learner $estimator,
        Dataset $training,
        Labeled $testing,
        Metric $metric
    ) {
        parent::__construct([self::class, 'score'], [
            $estimator, $training, $testing, $metric,
        ]);
    }
}
