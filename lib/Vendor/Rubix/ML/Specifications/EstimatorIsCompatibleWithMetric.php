<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

use function get_class;
use function in_array;

class EstimatorIsCompatibleWithMetric
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Estimator $estimator
     * @param \Test\Vendor\Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \InvalidArgumentException
     */
    public static function check(Estimator $estimator, Metric $metric) : void
    {
        if (!in_array($estimator->type(), $metric->compatibility())) {
            throw new InvalidArgumentException(
                Params::shortName(get_class($metric))
                    . ' is only compatible with '
                    . implode(', ', $metric->compatibility())
                    . " estimator types, {$estimator->type()} given."
            );
        }
    }
}
