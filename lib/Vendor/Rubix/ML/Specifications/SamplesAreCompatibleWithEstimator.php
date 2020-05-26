<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use InvalidArgumentException;

use function count;
use function get_class;

class SamplesAreCompatibleWithEstimator
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @param \Test\Vendor\Rubix\ML\Estimator $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Estimator $estimator) : void
    {
        $compatibility = $estimator->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($estimator))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
