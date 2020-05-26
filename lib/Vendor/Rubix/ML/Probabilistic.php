<?php

namespace Test\Vendor\Rubix\ML;

use Test\Vendor\Rubix\ML\Datasets\Dataset;

interface Probabilistic extends Estimator
{
    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public function proba(Dataset $dataset) : array;

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @param (string|int|float)[] $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array;
}
