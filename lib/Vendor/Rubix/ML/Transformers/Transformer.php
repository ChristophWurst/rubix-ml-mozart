<?php

namespace Test\Vendor\Rubix\ML\Transformers;

interface Transformer
{
    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array;

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void;
}
