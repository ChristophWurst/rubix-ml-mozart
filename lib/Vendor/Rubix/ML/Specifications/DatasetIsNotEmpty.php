<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class DatasetIsNotEmpty
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset) : void
    {
        if ($dataset->empty()) {
            throw new InvalidArgumentException('Dataset must contain'
                . ' at least one record.');
        }
    }
}
