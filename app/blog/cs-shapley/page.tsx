import Image from "next/image";
import katex from "katex";
import "katex/dist/katex.min.css";

export const metadata = {
  title: "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification – Stephanie Schoch",
};

function Tex({ math, display = false }: { math: string; display?: boolean }) {
  const html = katex.renderToString(math, { displayMode: display, throwOnError: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function CSShapleyBlog() {
  return (
    <article className="max-w-3xl mx-auto px-4 py-12 text-sm leading-relaxed">
      <h1 className="text-2xl font-bold mb-2">
        CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification
      </h1>
      <p className="text-text-light mb-1">Stephanie Schoch</p>
      <p className="text-text-light mb-8">November 28, 2022</p>

      <p className="text-xs text-text-light italic mb-8">
        A blog post about the NeurIPS 2022 paper &ldquo;CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification&rdquo;
      </p>

      <h2 className="text-xl font-semibold mt-8 mb-3">Data Contribution Estimation</h2>
      <p className="mb-4">
        In machine learning settings, there are notable benefits of understanding how individual training instances impact a learning model. For example, through identifying and filtering points that harm the model (e.g. noisy or mislabeled instances), the performance on a subsequent model retraining may increase. We could additionally seek to augment the data by identifying new data instances that are similar to training instances that were highly beneficial to the model. In this setting, we can refer to how an instance impacts a performance metric of choice as the &ldquo;contribution&rdquo; of the data point.
      </p>
      <p className="mb-4">
        The prudent question to ask then becomes one of how to measure this contribution. While we could simply measure the model performance when trained with vs. without the training instance (i.e. Leave-One-Out from Cook, 1977), this method has certain drawbacks as it does not satisfy several properties desirable for measuring data contributions and does not always perform as expected in practice. Ghorbani et al. (2019) exemplified this with a concrete example: if we are measuring contribution to a KNN classifier and have two copies of each data point, removal of one point would not change the classifier performance, and each data point would receive a contribution score of <Tex math="0" />.
      </p>

      <h3 className="text-lg font-semibold mt-6 mb-3">Shapley Values</h3>
      <p className="mb-4">
        Shapley values (Shapley, 1953) have been proposed for use in this context and have proven to be effective for measuring data contributions, and the associated applications. Shapley values, from cooperative game theory, satisfy desirable fairness guarantees due to their underlying axiomatic basis. For a value function <Tex math="v(\cdot)" />, the Shapley value <Tex math="\phi_i(T, \mathcal{A}, v)" />, for any data point <Tex math="i" /> is defined as:
      </p>
      <div className="my-6 overflow-x-auto text-center">
        <Tex
          display
          math="\phi_i(T, \mathcal{A}, v)= \sum_{S \subseteq T\setminus\{i\}} \frac{v(S\cup\{i\})-v(S)}{\binom{n-1}{|S|}}"
        />
      </div>
      <p className="mb-4">
        In simple terms, the Shapley value of a data point measures its average marginal contribution to every possible data subset.
      </p>
      <p className="mb-4">
        Much of the work in applying Shapley values to data contribution measurement, or data valuation, has sought to develop approximation techniques to mitigate the computational cost of true Shapley computation. Specifically, true Shapley value computation is exponential with respect to the number of data points, and as such, entails an exponential number of model retrainings. One such approximation method is the Truncated Monte Carlo method proposed by Ghorbani et al. (2019), which we adopt in this work. Additional approximation methods can be found listed in the paper.
      </p>

      <h2 className="text-xl font-semibold mt-8 mb-3">CS-Shapley</h2>

      <h3 className="text-lg font-semibold mt-6 mb-3">Intuition</h3>
      <p className="mb-4">
        What the existing methods have in common, is how the value function underlying Shapley computation is defined. More specifically, the value function is defined over the entire development set (in practice, development accuracy). In this work, we challenge the implicit assumption that full development set metrics are ideal for Shapley computation on classification datasets. Our intuition was that defining the value function in this manner may have limited ability to differentiate helpful or harmful training instances. We provide an example in Figure 1 below.
      </p>
      <figure className="my-6">
        <Image
          src="/cs-shapley-fig-1.png"
          alt="CS-Shapley Figure 1: Example showing two training points from CIFAR10 with same overall accuracy change but different in-class accuracy effects"
          width={380}
          height={220}
          className="mx-auto"
        />
      </figure>
      <p className="mb-4">
        While we provide more details in the paper, in short, this example shows two training points from the real world CIFAR10 datasets that belong to the same class, cause the same overall development accuracy change, yet data point <Tex math="i" /> increases in-class accuracy while data point <Tex math="j" /> decreases in-class accuracy. Intuitively, data points that harm their own classes may be mislabeled or otherwise noisy.
      </p>

      <h3 className="text-lg font-semibold mt-6 mb-3">CS-Shapley Definition</h3>
      <p className="mb-4">
        To address this, we define a value function that uses in-class accuracy as a measure of contribution and out-of-class accuracy as a weighting, or discounting, factor.
      </p>
      <p className="mb-4">
        Formally, we define the value function <Tex math="v(\cdot)" /> as
      </p>
      <div className="my-6 overflow-x-auto text-center">
        <Tex
          display
          math="v_{y_i}(S_{y_i}|S_{-y_i}) = a_S(D_{y_i})\cdot e^{a_S(D_{-y_i})}"
        />
      </div>
      <p className="mb-4">
        where <Tex math="a_S(D_{y_i})" /> indicates in-class accuracy and <Tex math="a_S(D_{-y_i})" /> indicates out-of-class accuracy. While we demonstrate several desirable properties of this function in the paper, we can illustrate this function in the following contour plot:
      </p>
      <figure className="my-6">
        <Image
          src="/fig-cd-contourplot.png"
          alt="Contour plot showing the CS-Shapley value function"
          width={380}
          height={220}
          className="mx-auto"
        />
      </figure>
      <p className="mb-4">
        The effect of the out-of-class accuracy is controlled by the value of the in-class accuracy. In other words, when the in-class accuracy is low, the out-of-class accuracy can essentially be ignored. Conversely, when the in-class accuracy is high, the out-of-class accuracy can have a substantial effect on the valuation of an in-class data point.
      </p>
      <p className="mb-4">
        With this, we can then define the <strong>CS-Shapley value</strong> of a data point <Tex math="i" /> as
      </p>
      <div className="my-6 overflow-x-auto text-center">
        <Tex
          display
          math="\phi_i|S_{-y_i} = \sum_{S_{y_i} \subseteq T_{y_i} \setminus \{i\}} \frac{v_{y_i}(S_{y_i}\cup\{i\}| S_{-y_i})-v_{y_i}(S_{y_i}| S_{-y_i})}{\binom{n-1}{|S_{y_i}|}}"
        />
      </div>
      <p className="mb-4">
        In{" "}
        <a
          href="https://arxiv.org/pdf/2211.06800.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent underline hover:text-accent/80"
        >
          the paper
        </a>
        , we demonstrate the efficacy of CS-Shapley and the underlying class-wise value function using three tasks: high-value data removal, noisy data detection, and transferability of data values. Please see our paper for more details!
      </p>

      <h2 className="text-xl font-semibold mt-8 mb-3">References</h2>
      <ul className="list-none space-y-2 text-text-light">
        <li>
          R Dennis Cook. Detection of influential observation in linear regression. <em>Technometrics</em>, 19(1):15&ndash;18, 1977.
        </li>
        <li>
          Amirata Ghorbani and James Zou. Data Shapley: Equitable valuation of data for machine learning. In <em>International Conference on Machine Learning</em>, pages 2242&ndash;2251. PMLR, 2019.
        </li>
        <li>
          Lloyd S Shapley. A value for n-person games, contributions to the theory of games, 2, 307&ndash;317, 1953.
        </li>
      </ul>
    </article>
  );
}
