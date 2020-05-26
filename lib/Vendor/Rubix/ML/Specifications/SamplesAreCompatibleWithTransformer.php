<?php

namespace Test\Vendor\Rubix\ML\Specifications;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use Test\Vendor\Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;

use function count;
use function get_class;

class SamplesAreCompatibleWithTransformer
{
    /**
     * Perform a check of the specification.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @param \Test\Vendor\Rubix\ML\Transformers\Transformer $transformer
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Transformer $transformer) : void
    {
        $compatibility = $transformer->compatibility();
        
        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($transformer))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
