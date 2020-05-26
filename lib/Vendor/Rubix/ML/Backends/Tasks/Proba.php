<?php

namespace Test\Vendor\Rubix\ML\Backends\Tasks;

use Test\Vendor\Rubix\ML\Probabilistic;
use Test\Vendor\Rubix\ML\Datasets\Dataset;

class Proba extends Task
{
    /**
     * Return the probabilities outputted by the estimator.
     *
     * @param \Test\Vendor\Rubix\ML\Probabilistic $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public static function proba(Probabilistic $estimator, Dataset $dataset) : array
    {
        return $estimator->proba($dataset);
    }

    /**
     * @param \Test\Vendor\Rubix\ML\Probabilistic $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Probabilistic $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'proba'], [$estimator, $dataset]);
    }
}
