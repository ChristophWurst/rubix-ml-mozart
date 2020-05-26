<?php

namespace Test\Vendor\Rubix\ML\Graph\Nodes;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use InvalidArgumentException;

use function count;

use const Test\Vendor\Rubix\ML\PHI;

/**
 * Isolator
 *
 * Isolator nodes represent splits in a tree designed to isolate groups
 * into cells by randomly dividing them.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Isolator implements BinaryNode
{
    use HasBinaryChildren;
    
    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected $column;

    /**
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right splits of the training data.
     *
     * @var \Test\Vendor\Rubix\ML\Datasets\Dataset[]
     */
    protected $groups;

    /**
     * Factory method to build a isolator node from a dataset
     * using a random split of the dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function split(Dataset $dataset) : self
    {
        $column = rand(0, $dataset->numColumns() - 1);

        $values = $dataset->column($column);

        if ($dataset->columnType($column)->isContinuous()) {
            $min = (int) floor(min($values) * PHI);
            $max = (int) ceil(max($values) * PHI);

            $value = rand($min, $max) / PHI;
        } else {
            $value = $values[array_rand($values)];
        }

        $groups = $dataset->partitionByColumn($column, $value);

        return new self($column, $value, $groups);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset[] $groups
     * @throws \InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups)
    {
        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of groups'
                . ' must be exactly 2.');
        }

        $this->column = $column;
        $this->value = $value;
        $this->groups = $groups;
    }

    /**
     * Return the feature column (index) of the split value.
     *
     * @return int
     */
    public function column() : int
    {
        return $this->column;
    }

    /**
     * Return the split value.
     *
     * @return int|float|string
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * Return the left and right splits of the training data.
     *
     * @return \Test\Vendor\Rubix\ML\Datasets\Dataset[]
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
