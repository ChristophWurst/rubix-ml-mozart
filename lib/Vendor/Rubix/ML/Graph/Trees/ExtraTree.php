<?php

namespace Test\Vendor\Rubix\ML\Graph\Trees;

use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Graph\Nodes\Comparison;

use function array_slice;

use const Test\Vendor\Rubix\ML\PHI;

/**
 * Extra Tree
 *
 * The base implementation of an *Extremely Randomized* Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class ExtraTree extends CART
{
    /**
     * Randomized algorithm that chooses the split point with the lowest impurity
     * among a random selection of features.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @return \Test\Vendor\Rubix\ML\Graph\Nodes\Comparison
     */
    protected function split(Labeled $dataset) : Comparison
    {
        shuffle($this->columns);

        $columns = array_slice($this->columns, 0, $this->maxFeatures);

        $bestImpurity = INF;
        $bestColumn = 0;
        $bestValue = null;
        $bestGroups = [];

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            if ($this->types[$column]->isContinuous()) {
                $min = (int) floor(min($values) * PHI);
                $max = (int) ceil(max($values) * PHI);
    
                $value = rand($min, $max) / PHI;
            } else {
                $values = array_unique($values);
                
                $value = $values[array_rand($values)];
            }

            $groups = $dataset->partitionByColumn($column, $value);

            $impurity = $this->splitImpurity($groups);

            if ($impurity < $bestImpurity) {
                $bestColumn = $column;
                $bestValue = $value;
                $bestGroups = $groups;
                $bestImpurity = $impurity;
            }

            if ($impurity <= self::IMPURITY_TOLERANCE) {
                break 1;
            }
        }

        return new Comparison(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity
        );
    }
}
