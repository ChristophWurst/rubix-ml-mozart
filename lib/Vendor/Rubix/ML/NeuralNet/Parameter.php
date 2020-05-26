<?php

namespace Test\Vendor\Rubix\ML\NeuralNet;

use TEST_Tensor\Tensor;

class Parameter
{
    /**
     * The auto incrementing id.
     *
     * @var int
     */
    protected static $counter = 0;

    /**
     * The unique identifier of the parameter.
     *
     * @var int
     */
    protected $id;

    /**
     * The parameter.
     *
     * @var \TEST_Tensor\Tensor
     */
    protected $param;

    /**
     * @param \TEST_Tensor\Tensor $param
     */
    public function __construct(TEST_Tensor $param)
    {
        $this->id = self::$counter++;
        $this->param = $param;
    }

    /**
     * Return the unique identifier of the parameter.
     *
     * @return int
     */
    public function id() : int
    {
        return $this->id;
    }

    /**
     * Return the wrapped parameter.
     *
     * @return mixed
     */
    public function param()
    {
        return $this->param;
    }

    /**
     * Update the parameter.
     *
     * @param \TEST_Tensor\Tensor $step
     */
    public function update(TEST_Tensor $step) : void
    {
        $this->param = $this->param->subtract($step);
    }

    /**
     * Perform a deep copy of the object upon cloning.
     */
    public function __clone()
    {
        $this->param = clone $this->param;
    }
}
