<?php

namespace Test\Vendor\Rubix\ML\CrossValidation;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric;

interface Validator
{
    /**
     * Test the estimator with the supplied dataset and return a validation score.
     *
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float;
}
