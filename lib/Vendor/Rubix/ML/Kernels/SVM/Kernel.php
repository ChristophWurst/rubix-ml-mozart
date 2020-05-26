<?php

namespace Test\Vendor\Rubix\ML\Kernels\SVM;

interface Kernel
{
    /**
     * Return the options for the libsvm runtime.
     *
     * @return mixed[]
     */
    public function options() : array;
}
