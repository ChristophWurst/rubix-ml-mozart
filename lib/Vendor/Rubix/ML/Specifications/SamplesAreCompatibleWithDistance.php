<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use Test\Vendor\Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

use function count;
use function get_class;

class SamplesAreCompatibleWithDistance
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @param \Test\Vendor\Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Distance $kernel) : void
    {
        $compatibility = $kernel->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($kernel))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
