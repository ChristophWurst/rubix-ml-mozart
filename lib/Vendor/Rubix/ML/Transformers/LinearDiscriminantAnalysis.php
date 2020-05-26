<?php

namespace Test\Vendor\Rubix\ML\Transformers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function array_slice;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Linear Discriminant Analysis
 *
 * Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that
 * selects the most informative features based on their class labels. More formally, LDA finds
 * a linear combination of features that characterizes or best *discriminates* two or more
 * classes.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LinearDiscriminantAnalysis implements Transformer, Stateful
{
    /**
     * The target number of dimensions to project onto.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The matrix of eigenvectors computed at fitting.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $eigenvectors;

    /**
     * The amount of variance that is preserved by the transformation.
     *
     * @var float|null
     */
    protected $explainedVar;

    /**
     * The amount of variance lost by discarding the noise components.
     *
     * @var float|null
     */
    protected $noiseVar;

    /**
     * The percentage of information lost due to the transformation.
     *
     * @var float|null
     */
    protected $lossiness;

    /**
     * @param int $dimensions
     * @throws \InvalidArgumentException
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        $this->dimensions = $dimensions;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->eigenvectors);
    }

    /**
     * Return the amount of variance that has been preserved by the
     * transformation.
     *
     * @return float|null
     */
    public function explainedVar() : ?float
    {
        return $this->explainedVar;
    }

    /**
     * Return the amount of variance lost by discarding the noise components.
     *
     * @return float|null
     */
    public function noiseVar() : ?float
    {
        return $this->noiseVar;
    }

    /**
     * Return the percentage of information lost due to the transformation.
     *
     * @return float|null
     */
    public function lossiness() : ?float
    {
        return $this->lossiness;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        if ($dataset->labelType() != DataType::categorical()) {
            throw new InvalidArgumentException('Transformer requires'
                . " categorical labels, {$dataset->labelType()} given.");
        }

        [$m, $n] = $dataset->shape();

        $sW = TEST_Matrix::zeros($n, $n);

        foreach ($dataset->stratify() as $stratum) {
            $sW = TEST_Matrix::build($stratum->samples())
                ->transpose()
                ->covariance()
                ->multiply($stratum->numRows() / $m)
                ->add($sW);
        }

        $eig = TEST_Matrix::quick($dataset->samples())
            ->transpose()
            ->covariance()
            ->subtract($sW)
            ->eig(true);

        $eigenvalues = $eig->eigenvalues();

        $eigenvectors = $eig->eigenvectors()->asArray();

        $totalVar = array_sum($eigenvalues);
        
        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        $eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
        $eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);

        $eigenvectors = TEST_Matrix::quick($eigenvectors)->transpose();

        $explainedVar = (float) array_sum($eigenvalues);
        $noiseVar = $totalVar - $explainedVar;

        $this->explainedVar = $explainedVar;
        $this->noiseVar = $noiseVar;
        $this->lossiness = $noiseVar / ($totalVar ?: EPSILON);

        $this->eigenvectors = $eigenvectors;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->eigenvectors) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = TEST_Matrix::build($samples)
            ->matmul($this->eigenvectors)
            ->asArray();
    }
}
