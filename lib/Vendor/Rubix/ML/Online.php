<?php

namespace Test\Vendor\Rubix\ML;

use Test\Vendor\Rubix\ML\Datasets\Dataset;

interface Online extends Learner
{
    /**
     * Perform a partial train on the learner.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void;
}
