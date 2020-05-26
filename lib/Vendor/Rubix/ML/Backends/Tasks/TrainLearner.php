<?php

namespace Test\Vendor\Rubix\ML\Backends\Tasks;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Datasets\Dataset;

class TrainLearner extends Task
{
    /**
     * Train a learner and return the instance.
     *
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return \Test\Vendor\Rubix\ML\Learner
     */
    public static function train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Learner $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'train'], [$estimator, $dataset]);
    }
}
