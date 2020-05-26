<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class LabelsAreCompatibleWithLearner
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @param \Test\Vendor\Rubix\ML\Learner $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Labeled $dataset, Learner $estimator) : void
    {
        $labelType = $dataset->labelType();

        switch ($estimator->type()) {
            case EstimatorType::classifier():
                if ($labelType != DataType::categorical()) {
                    throw new InvalidArgumentException('Classifiers require'
                        . " categorical labels, $labelType given.");
                }

                break 1;

            case EstimatorType::regressor():
                if ($labelType != DataType::continuous()) {
                    throw new InvalidArgumentException('Regressors require'
                        . " continuous labels, $labelType given.");
                }

                break 1;
        }
    }
}
