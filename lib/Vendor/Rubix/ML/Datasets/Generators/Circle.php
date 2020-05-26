<?php

namespace Test\Vendor\Rubix\ML\Datasets\Generators;

use TEST_Tensor\TEST_Matrix;
use TEST_Tensor\TEST_Vector;
use TEST_Tensor\TEST_ColumnVector;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

use const Test\Vendor\Rubix\ML\TWO_PI;

/**
 * Circle
 *
 * Create a circle made of sample data points in 2 dimensions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Circle implements Generator
{
    /**
     * The center vector of the circle.
     *
     * @var \TEST_Tensor\TEST_Vector
     */
    protected $center;

    /**
     * The scaling factor of the circle.
     *
     * @var float
     */
    protected $scale;

    /**
     * The factor of gaussian noise to add to the data points.
     *
     * @var float
     */
    protected $noise;

    /**
     * @param float $x
     * @param float $y
     * @param float $scale
     * @param float $noise
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $x = 0.0,
        float $y = 0.0,
        float $scale = 1.0,
        float $noise = 0.1
    ) {
        if ($scale < 0.0) {
            throw new InvalidArgumentException('Scale must be'
                . " greater than 0, $scale given.");
        }

        if ($noise < 0.0) {
            throw new InvalidArgumentException('Noise must be'
                . " greater than 0, $noise given.");
        }

        $this->center = TEST_Vector::quick([$x, $y]);
        $this->scale = $scale;
        $this->noise = $noise;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return 2;
    }

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Test\Vendor\Rubix\ML\Datasets\Labeled
     */
    public function generate(int $n) : Labeled
    {
        $r = TEST_ColumnVector::rand($n)->multiply(TWO_PI);

        $x = $r->cos();
        $y = $r->sin();

        $noise = TEST_Matrix::gaussian($n, 2)
            ->multiply($this->noise);

        $samples = TEST_Matrix::stack([$x, $y])
            ->multiply($this->scale)
            ->add($this->center)
            ->add($noise)
            ->asArray();

        $labels = $r->rad2deg()->asArray();

        return Labeled::quick($samples, $labels);
    }
}
