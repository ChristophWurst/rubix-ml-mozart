<?php

namespace Test\Vendor\Rubix\ML\Backends\Tasks;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Datasets\Dataset;

class Predict extends Task
{
    /**
     * Return the predictions outputted by an estimator.
     *
     * @param \Test\Vendor\Rubix\ML\Estimator $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return (int|float|string)[]
     */
    public static function predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }

    /**
     * @param \Test\Vendor\Rubix\ML\Estimator $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Estimator $estimator, Dataset $dataset)
    {
        parent::__construct([self::class, 'predict'], [$estimator, $dataset]);
    }
}
