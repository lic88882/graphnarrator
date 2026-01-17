import json
import os
import time
from pathlib import Path
from typing import Literal, Union

from openai import OpenAI

from log import logger
from tag import TAG


class TemplatesArchive:
    templates = {
        "v1": """A graph-neural-network model has classified a graph's node as belonging to the "{label}" category from the seven possible categories (['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']). 
The higher score behind the token (features of node) indicates greater importance for the prediction, and vice versa. The prediction was not based on these scores. Rather, it was based on the target paper ([ROOT]) itself and papers that cited the [ROOT]. 
The structure of the following document is converted from the citation subgraph. Each row starts with a number (if not [ROOT]), representing the unique identifier of a paper, which also indicates the specific citation relationship (e.g., x.y.z means z cited y, and y cited x). 
<document>{document}</document>
Based on the importance scores of the target paper and other papers that cited the target paper, briefly explain why the GNN model predicted the target as "{label}" category.""",
        "v2": """A graph-neural-network (GNN) model has classified a graph's node as belonging to the "{label}" category from the seven possible categories (['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']).
<document>{document}</document>
The document above represents the citation relationships among papers. Each paper's unique identifier shows how papers cite each other, with the root paper being the target. The scores behind each token indicate the importance of the node's features.
Now, please tell a story based on the importance scores of the target paper and other papers that cited the target paper. Explain why the GNN model predicted the target paper as belonging to the {label} category.""",
        "v3": """A graph-neural-network (GNN) model has classified a graph's node as belonging to the "{label}" category from the seven possible categories (['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']).
<document>{document}</document>
The document above represents the citation relationships among papers. Each paper's unique identifier shows how papers cite each other, with the root paper being the target.
Now, based on the keywords of the target paper and other papers that cited the target paper, try to explain why the GNN model predicted the target paper as belonging to the {label} category.""",
        "v4": """The following jsonified graph contains important words in the text of each nodes. These words contributes to the classification of Node 0 into the "{label}" category from the seven possible categories (['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']). 
<JSONIFIED-Graph>{document}</JSONIFIED-Graph>
Try to compress the above JSON content into a concise human-readable free-text explanation, identify helpful inner-node and inter-node information in order to justify the classification result.""",
        "v5": """The following jsonified graph contains important words in the text of each nodes. These words contributes to the classification of Node 0 into one of the seven possible categories (['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']). 
Generate a concise, human-readable explanation that justifies the classification result of Node 0 by identifying and explaining the relevant inner-node features (i.e., keywords) and inter-node relationships (i.e., graph structure). The explanation should focus on how these factors contribute to the classification label.

## Example

### Jsonified Graph
<JSONIFIED-Graph>{example_verbalized_graph}</JSONIFIED-Graph>

### Classification Label
Probabilistic Methods

### Reasoning
0. Premise:
In the provided jsonified graph, there are 13 nodes in total. Node 0 is the target node for classification. The "neighbors" list of Node 0 contains ["1", "2"], indicating that Node 1 and Node 2 are direct neighbors of Node 0. Similarly, Nodes 3 to 8 are direct neighbors of Node 1, and Nodes 9 to 12 are direct neighbors of Node 2. The proximity of these nodes to Node 0 influences their relevance to the classification result. Generally, direct neighbors have a stronger influence on the classification of Node 0 compared to indirect neighbors, as the information from direct neighbors is more immediately accessible.

1. Semantic Inference:
Synonyms and related terms for the classification label, "Probabilistic Methods," are crucial in determining the classification. Keywords synonymous with or closely related to 'Probabilistic Methods'—such as 'probabilistic', 'probability', and 'inference'—are highly influential in determining the classification because they directly indicate the node's relevance to the 'Probabilistic Methods' category. In contrast, more general keywords like 'algorithm' and 'learning', while relevant to broader fields within AI, do not specifically point to 'Probabilistic Methods' and thus contribute less directly to the classification.

2. Combination:
When combining the graph structure with semantic analysis, the flow of relevant information from neighboring nodes to Node 0 becomes critical. Direct neighbors like Node 2, which contains the keyword 'probabilistic', have a substantial influence on Node 0's classification due to their close proximity and the relevance of their content. Although Node 1 does not contain as many directly relevant keywords as Node 2, it still discusses POMDPs, which are probabilistic models, indicating that it contributes to the classification, albeit in a more nuanced way. Indirect neighbors also play a role, though their impact diminishes with distance from Node 0. For instance, Nodes 9 to 12, connected through Node 2, contain keywords like 'probabilistic' and 'probability', which further reinforce the classification of Node 0. Meanwhile, the indirect neighbors connected through Node 1, such as Nodes 3 to 8, are less relevant because they do not emphasize keywords directly related to 'Probabilistic Methods'. However, their influence should not be entirely discounted, as they may still provide some contextual relevance through their connection to Node 1's discussion of probabilistic models.

### Free-Text Explanation
```markdown
The classification of Node 0 into the "Probabilistic Methods" category is strongly supported by both its internal features and its relationships with neighboring nodes.

- **Target Node**: The presence of keywords like "probabilistic" and "inference" within Node 0 suggests that the paper is focused on topics central to probabilistic methods, such as performing probabilistic inference and evaluating influence diagrams.

- **Direct Neighbors**: 
   - **Node 1**: While Node 1 focuses on POMDPs, which are also probabilistic models, it is slightly less directly related to the probabilistic inference methods emphasized in Node 0 and Node 2. However, it still contributes positively to the classification.
   - **Node 2**: This node discusses "probabilistic inference" and "Bayesian networks," both of which are foundational to probabilistic methods. The strong thematic connection between Node 0 and Node 2 reinforces the classification.

- **Indirect Neighbors**: 
   - Nodes connected through Node 1 are less relevant to probabilistic methods and thus contribute less to the classification.
   - Nodes connected through Node 2, such as Nodes 9, 10, 11, and 12, also discuss probabilistic topics, further supporting the classification, though their influence is more indirect. 

In summary, the classification of Node 0 into "Probabilistic Methods" is well-supported by the strong presence of key probabilistic terms within the node itself and its direct connection to Node 2, which also focuses on probabilistic inference. Indirect neighbors through Node 2 further reinforce this classification.
```

## Task

### Jsonified Graph
<JSONIFIED-Graph>{document}</JSONIFIED-Graph>

### Classification Label
{label}

### Reasoning

### Free-Text Explanation

(P.S.: 1. make sure to complete both the reasoning section and then Free-Text Explanation section with the same structure as exemplified above.
2. make sure to double-check the graph structure when generating free-text explanation. Specifically, distinguishing direct and indirect neighbors is important.)
""",
        "v6": """The following verbalized graph contains important words in the text of each nodes. These words (each with corresponding importance score in the bracket) contributes to the classification of Node 0 into one of the seven possible categories (['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']). 
Generate a concise, human-readable explanation that justifies the classification result of Node 0 by identifying and explaining the relevant inner-node features (i.e., keywords) and inter-node relationships (i.e., graph structure). The explanation should focus on how these factors contribute to the classification label.

## Example

### Verbalized Graph
<verbalized-graph>{example_verbalized_graph}</verbalized-graph>

### Classification Label
Probabilistic Methods

### Reasoning

0. Graph Structure Reconstruction:

In the provided verbalized graph, The ROOT node (first line) is the target for classification.
Single-digit indexed nodes are direct neighbors of ROOT.
Double-digit indexed nodes are:
  - Two hops away from ROOT
  - Direct children of their parent node
More digits indexed nodes follow the same principle as described above.

Thus, the graph structure of this verbalized graph is:

- ROOT
  - Node-1
    - Node-1.1
    - Node-1.2
    - Node-1.3
    - Node-1.4
    - Node-1.5
    - Node-1.6
  - Node-2
    - Node-2.1
    - Node-2.2
    - Node-2.3
    - Node-2.4

1. Word-Level Evaluation:

Detect important terms for the classification label.
Quantitatively, the importance (saliency) scores behind each word in the verbalized graph are calculated by a post-hoc XAI (explanable AI) algorithm.
Semantically, keywords synonymous with or closely related to label, 'Probabilistic Methods' (such as 'probabilistic', 'probability', and 'inference'), are sementically influential in determining the classification because they directly indicate the node's relevance to the 'Probabilistic Methods' category.

2. Graph-Level Aggregation:

Node 1 does not contain as many directly relevant keywords. Yet, it still discusses POMDPs, which are probabilistic models, indicating that it contributes to the classification in a less significant way. 
Direct neighbors like Node 2, which contains the keyword 'probabilistic', have a substantial influence on ROOT node classification due to their close proximity to the ROOT node.
Indirect neighbors, Nodes-2.1 to Node-2.4, connected through Node 2, contain keywords like 'probabilistic' and 'probability', which further reinforce the classification of ROOT. 
Meanwhile, the indirect neighbors connected through Node 1, Node-1.1 to Node-1.6, are less relevant because they do not include important keywords directly related to 'Probabilistic Methods'.

### Free-Text Explanation
```markdown
The classification of ROOT node into the "Probabilistic Methods" category can be explained as follows:

ROOT: The presence of keywords like "probabilistic" and "inference" within ROOT node suggests that the paper is focused on topics central to probabilistic methods, such as performing probabilistic inference and evaluating influence diagrams.
  - Node-1: While Node 1 focuses on POMDPs, which are also probabilistic models, it is slightly less directly related to the probabilistic inference methods emphasized in ROOT node.
    - Node-1.1 ~ Node-1.6: they are less relevant to probabilistic methods and thus contribute less to the classification.
  - Node-2: This node discusses "probabilistic inference" and "Bayesian networks," both of which are foundational to probabilistic methods. The strong thematic connection between ROOT node and Node 2 reinforces the classification.
    - Node-2.1 ~ Node-2.4: they also discuss probabilistic topics, further supporting the classification.

In summary, the classification of ROOT node into "Probabilistic Methods" is well-supported by the strong presence of key probabilistic terms within the node itself and its direct connection to Node 2, which also focuses on probabilistic inference. Indirect neighbors through Node 2 further reinforce this classification.
```

## Task

### Verbalized Graph
<verbalized-graph>{document}</verbalized-graph>

### Classification Label
{label}

### Reasoning

### Free-Text Explanation

(P.S.: 1. make sure to complete both the reasoning section and then Free-Text Explanation section with the same structure as exemplified above.
2. make good use of the importance (saliency) score behind each word as your guidance to generate the better explanation. However, it is not necessary to directly quote the saliency score.
3. use the *whole* graph structure you constructed during reasoning for the format of the explanation. Indents and node indexes are necessary, which representing the hierarchy of the graph.)
""",
    }

    @classmethod
    def get(cls, version: Union[Literal["latest"], str]):
        if version == "latest":
            return list(cls.templates.values())[-1]

        return cls.templates[version]


class ExampleGraphCorpus:
    example_jsonified_graph = {
        "v1": """{"0": {"neighbors": ["1", "2"], "node_features": "title : some experiments with real - time decision algorithms abstract : real - time decision algorithms are a class of incremental resource - bounded [ horvitz, 89 ] or anytime [ dean, 93 ] algorithms for evaluating influence diagrams. we present a test domain for real - time decision algorithms, and the results of experiments with several real - time decision algorithms in this domain. the results demonstrate high performance for two algorithms, a decision - evaluation variant of incremental probabilisitic inference [ dambrosio, 93 ] and a variant of an algorithm suggested by goldszmidt, [ goldszmidt, 95 ], pk - reduced. we discuss the implications of these experimental results and explore the broader applicability of these algorithms. "}, "1": {"neighbors": ["0", "3", "4", "5", "6", "7", "8"], "node_features": "title : learning policies for partially observable environments : scaling up abstract : partially observable markov decision processes ( pomdp ' s ) model decision problems in which an agent tries to maximize its reward in the face of limited and / or noisy sensor feedback. while the study of pomdp ' s is motivated by a need to address realistic problems, existing techniques for finding optimal behavior do not appear to scale well and have been unable to find satisfactory policies for problems with more than a dozen states. after a brief review of pomdp ' s, this paper discusses several simple solution methods and shows that all are capable of finding near - optimal policies for a selection of extremely small pomdp ' s taken from the learning literature. in contrast, we show that none are able to solve a slightly larger and noisier problem based on robot navigation. we find that a combination of two novel approaches performs well on these problems and suggest methods for scaling to even larger and more complicated domains. "}, "2": {"neighbors": ["0", "9", "10", "11", "12"], "node_features": "title : efficient inference in bayes networks as a combinatorial optimization problem abstract : a number of exact algorithms have been developed to perform probabilistic inference in bayesian networks in recent years. the techniques used in these algorithms are closely related to network structures and some of them are not easy to understand and implement. in this paper, we consider the problem from the combinatorial optimization point of view and state that efficient probabilistic inference in a belief network is a problem of finding an optimal factoring given a set of probability distributions. from this viewpoint, previously developed algorithms can be seen as alternate factoring strategies. in this paper, we define a combinatorial optimization problem, the optimal factoring problem, and discuss application of this problem in belief networks. we show that optimal factoring provides insight into the key elements of efficient probabilistic inference, and demonstrate simple, easily implemented algorithms with excellent performance. "}, "3": {"neighbors": [], "node_features": "title : a formal framework speedup learning from problems and solutions abstract : speedup learning seeks improve the computational efficiency problem solving with experience. in this paper, we develop formal framework learning efficient problem solving random problems solutions. we apply this framework two different representations of learned knowledge, namely control rules and macro - operators, and prove theorems identify conditions learning representation. our proofs are constructive in that accompanied learning algorithms. our framework captures both empirical and explanation - based speedup learning in a unified fashion. we illustrate our framework with implementations in two domains : symbolic integration and eight puzzle. this work integrates many strands of experimental and theoretical work in machine learning, including empirical learning of control rules, macro - operator learning, "}, "4": {"neighbors": [], "node_features": "title : acting under uncertainty : discrete bayesian models for mobile - robot navigation abstract : discrete bayesian models have been used to model uncertainty for mobile - robot navigation, but the question of how actions should be chosen remains largely unexplored. this paper presents the optimal solution to the problem, formulated as a partially observable markov decision process. since solving for the optimal control policy is intractable, in general, it goes on to explore a variety of heuristic control strategies. the control strategies are compared experimentally, both in simulation and in runs on a robot. "}, "5": {"neighbors": [], "node_features": "title : incremental methods for computing bounds in markov decision processes abstract : partially markov decision processes ( ) allow one to model complex dynamic decision or control problems that include both action outcome uncertainty and imperfect ity. the control problem is formulated as a dynamic optimization problem with value function combining costs or rewards from multiple steps. in this paper we propose, analyse and test various incremental methods computing bounds the for control problems infinite discounted horizon criteria. the methods described and tested include novel incremental versions of grid - based linear method and simple lower bound method sondik updates. both these can work with arbitrary points of the belief space and enhanced various heuristic point selection strategies. also introduced is a new method for computing an initial upper bound the fast informed bound method. this method able improve significantly standard commonly upper bound computed by - based method. the quality of resulting bounds are tested on a maze navigation problem with 20 states, 6 actions and 8 observations. "}, "6": {"neighbors": [], "node_features": "title : learning sorting and decision trees with pomdps abstract : pomdps are general models of sequential decisions which both actions and observations probabilistic. many problems of interest can be formulated as pomdps, yet the use pomdps has limited the lack effective algorithms. recently this started change problems such as robot navigation planning beginning formulated and solved as pomdps. the advantage of the pomdp approach is its clean semantics and ability principled solutions integrate physical and information gathering actions. in this paper we pursue this approach in the context of two learning tasks : learning to sort a vector numbers and learning decision trees from data. both problems are formulated as pomdps and solved by a general pomdp algorithm. the main lessons and results are that 1 ) the use suitable heuristics representations allows solution sorting and classification pomdps of trivial sizes, 2 ) quality resulting solutions competitive best algorithms, and 3 ) problematic aspects in decision tree learning such test and mis classification costs, noisy tests, and missing values are naturally accommodated. "}, "7": {"neighbors": [], "node_features": "title : approximating optimal policies for partially observable stochastic domains abstract : the problem of optimal decisions in uncertain conditions is central to artificial intelligence. if state the world known all world modeled markov decision process mdps studied extensively and many methods known determining optimal courses action, or policies. the more realistic case where state information is only partially observable, partially observable markov decision processes ( ), have received much less attention. the best exact algorithms for these problems can be very inefficient space time. we introduce smooth partially observable value approximation ( spova ), a new approximation method that can quickly yield good approximations improve time. this method can be combined with reinforcement learning methods, a combination that was very effective in our test cases. "}, "8": {"neighbors": [], "node_features": "title : efficient dynamic - programming updates in partially observable markov decision processes abstract : we examine the problem of performing exact dynamic - programming updates in partially observable markov decision processes ( pomdps ) from a computational complexity viewpoint. dynamic - programming updates are a crucial operation in a wide range of pomdp solution methods and we find that it is intractable to perform these updates on piecewise - linear convex value functions for general pomdps. we offer a new algorithm, called the witness algorithm, which can compute updated value functions efficiently on a restricted class of pomdps in which the number of linear facets is not too great. we compare the witness algorithm to existing algorithms analytically and empirically and find that it is the fastest algorithm over a wide range of pomdp sizes. "}, "9": {"neighbors": [], "node_features": "title : sensitivities : alternative conditional probabilities for bayesian belief networks abstract : we show an alternative way representing a bayesian belief network by sensitivities and probability distributions. this representation equivalent traditional representation conditional probabilities, dependencies between nodes apparent and intuitively easy to understand. we also propose a qr matrix representation the sensitivities conditional efficient, memory requirements computational speed, than the traditional representation for computer implementations inference. we use sensitivities to show that for a certain class of binary networks, the computation time for approximate probabilistic inference upper bound error result is independent size the network. finally, as an alternative to traditional algorithms that use conditional probabilities, we describe an exact algorithm for probabilistic inference that uses the qr - representation for sensitivities and updates probability distributions of nodes in a network according to messages from the neigh bors. "}, "10": {"neighbors": [], "node_features": "title : algebraic techniques efficient inference in bayesian networks abstract : a number of exact algorithms have been developed perform probabilistic inference in bayesian belief networks in recent years. these algorithms use graph - techniques analyze and exploit network topology. in this paper, we examine the problem efficient probabilistic inference in a belief network as a combinatorial optimization problem, that finding an optimal factoring given an algebraic expression over a of probability distributions. we define a combinatorial optimization problem, the optimal factoring problem, and discuss application this problem in belief networks. we show that optimal factoring provides insight the elements of efficient probabilistic inference, and present simple, easily implemented algorithms with excellent performance. we also show how use of an algebraic perspective permits significant extension to the belief net representation. "}, "11": {"neighbors": [], "node_features": "title : interpretation of complex scenes using bayesian networks abstract : in most object recognition systems, interactions between objects a scene ignored and the best interpretation the objects the greatest image features. we show how image interpretation can be cast as the problem of finding the probable explanation ( ) in a bayesian network models visual and physical object interactions. the problem of how determine exact conditional probabilities for the network is shown to be unimportant, since the goal find probable configuration objects, calculate absolute probabilities. we furthermore show that evaluating configurations by feature counting is equivalent to calculating joint probability the configuration using restricted bayesian network, and derive the assumptions about probabilities necessary a bayesian formulation reasonable. "}, "12": {"neighbors": [], "node_features": "title : case - based probability factoring in bayesian belief networks abstract : bayesian network inference can formulated a combinatorial optimization problem, concerning in the computation of optimal factoring for the distribution represented in the net. since determination optimal is computationally hard problem, heuristic greedy strategies able approximations the optimal usually adopted. in the present paper we investigate an alternative approach based on a combination of genetic algorithms ( ga ) and case - based reasoning ( cbr ). we show how the use of genetic algorithms can improve the quality the computed factoring in case static strategy used ( as the mpe computation ), while the combination of ga and cbr can still provide advantages in case of dynamic strategies. some preliminary results on different kinds of nets are then reported. "}}"""
    }
    example_graph = {
        "v1": """ROOT: title(9.13) experiments(7.56) real(2.52) time(2.41) decision(5.20) algorithms(7.18) abstract(12.01) real(3.17) time(2.82) decision(5.46) algorithms(10.39) class(4.34) incremental(2.60) resource(4.50) bounded(5.79) horvitz,(2.67) 89(4.58) anytime(6.66) dean,(4.92) 93(5.03) algorithms(7.94) evaluating(4.75) influence(7.70) diagrams.(10.34) present(16.50) test(6.61) domain(10.50) real(3.11) time(2.89) decision(5.51) algorithms,(5.84) results(6.80) experiments(7.37) several(2.94) real(1.83) time(1.94) decision(4.34) algorithms(5.16) domain.(8.45) results(10.73) demonstrate(14.65) high(5.51) performance(6.46) two(7.53) algorithms,(6.19) decision(4.79) evaluation(4.18) variant(3.69) incremental(2.25) probabilisitic(2.59) inference(6.22) dambrosio,(3.81) 93(4.42) variant(3.46) algorithm(5.22) suggested(3.74) goldszmidt,(3.28) goldszmidt,(2.38) 95(5.24) ],(3.08) pk(2.77) reduced.(6.20) discuss(13.66) implications(9.36) experimental(9.82) results(9.38) explore(8.79) broader(5.65) applicability(3.44) algorithms.(14.64) 
Node-1: title(12.47) learning(12.87) policies(9.77) partially(3.11) observable(2.82) environments(5.58) scaling(9.39) abstract(10.80) partially(4.42) observable(2.62) markov(4.50) decision(5.75) processes(4.53) pomdp(9.69) model(11.47) decision(7.63) problems(7.18) agent(12.00) tries(3.13) maximize(3.05) reward(6.03) face(2.13) limited(2.17) noisy(8.96) sensor(6.27) feedback.(5.17) study(4.64) pomdp(15.31) motivated(4.24) need(2.26) address(2.23) realistic(4.37) problems,(2.86) existing(3.55) techniques(5.16) finding(4.80) optimal(15.47) behavior(5.35) appear(2.47) scale(3.92) well(2.03) unable(3.29) find(3.00) satisfactory(4.62) policies(7.82) problems(4.45) dozen(4.76) states.(10.45) brief(5.22) review(6.37) pomdp(6.59) s,(4.16) paper(10.26) discusses(7.55) several(5.13) simple(4.01) solution(4.55) methods(5.11) shows(4.13) capable(5.08) finding(4.90) near(3.28) optimal(21.93) policies(18.79) selection(5.62) extremely(5.44) small(4.79) pomdp(23.78) taken(4.85) learning(17.53) literature.(7.03) contrast,(4.73) show(3.74) none(6.38) able(2.67) solve(3.53) slightly(3.31) larger(3.38) noisier(3.47) problem(4.47) based(2.61) robot(20.05) navigation.(10.29) find(4.77) combination(3.98) two(3.96) novel(8.04) approaches(5.12) performs(3.93) well(2.61) problems(5.02) suggest(7.78) methods(5.07) scaling(10.48) even(2.50) larger(2.94) complicated(4.88) domains.(8.93) 
Node-1.1: title(0.95) formal(0.36) framework(0.41) speedup(0.35) learning(0.41) problems(0.48) solutions(0.48) abstract(1.14) speedup(0.33) learning(0.61) seeks(0.57) improve(0.27) computational(0.50) efficiency(0.35) problem(0.37) solving(0.41) experience.(0.57) paper,(0.70) develop(0.53) formal(0.40) framework(0.38) learning(0.37) efficient(0.40) problem(0.34) solving(0.32) random(0.54) problems(0.32) solutions.(0.37) apply(0.42) framework(0.47) two(0.45) different(0.24) representations(0.54) learned(0.64) knowledge,(0.41) namely(0.58) control(0.99) rules(0.73) macro(0.54) operators,(0.48) prove(0.58) theorems(0.38) identify(0.28) sufficient(0.22) conditions(0.28) learning(0.38) representation.(0.46) proofs(0.50) constructive(0.81) accompanied(0.47) learning(0.54) algorithms.(0.65) framework(0.83) captures(1.22) empirical(0.63) explanation(0.81) based(0.44) speedup(0.39) learning(0.67) unified(0.80) fashion.(0.54) illustrate(1.50) framework(0.56) implementations(0.79) two(0.46) domains(0.76) symbolic(0.78) integration(0.70) eight(0.65) puzzle.(0.81) work(1.09) integrates(0.74) many(0.54) strands(1.11) experimental(0.75) theoretical(0.55) work(0.49) machine(0.99) learning,(0.66) including(0.61) empirical(0.82) learning(0.59) control(0.85) rules,(0.83) macro(0.81) operator(1.00) learning,(1.31) 
Node-1.2: title(2.30) acting(0.98) uncertainty(2.31) discrete(1.03) bayesian(1.13) models(0.94) mobile(1.03) robot(1.75) navigation(1.12) abstract(2.80) discrete(1.18) bayesian(0.97) models(0.81) used(0.66) model(0.56) uncertainty(1.79) mobile(0.77) robot(1.55) navigation,(0.66) question(0.64) actions(1.17) chosen(0.67) remains(0.72) largely(0.56) unexplored.(0.74) paper(3.63) presents(1.59) optimal(1.17) solution(1.03) problem,(1.09) formulated(0.96) partially(0.40) observable(0.42) markov(0.90) decision(0.74) process.(0.85) since(1.13) solving(1.17) optimal(1.16) control(0.75) policy(1.02) intractable,(0.74) general,(0.82) goes(1.52) explore(2.04) variety(1.42) heuristic(0.84) control(1.05) strategies.(2.28) control(1.40) strategies(3.63) compared(3.99) experimentally,(1.72) simulation(2.58) runs(1.63) robot.(3.17) 
Node-1.3: title(1.50) incremental(0.64) methods(0.50) computing(0.59) bounds(1.09) partially(0.24) observable(0.21) markov(0.31) decision(0.64) processes(0.38) abstract(0.97) partially(0.25) observable(0.21) markov(0.32) decision(0.58) processes(0.36) pomdps(0.21) allow(0.54) one(0.36) model(0.47) complex(0.38) dynamic(0.60) decision(0.76) control(0.38) problems(0.53) include(0.31) action(0.82) outcome(0.55) uncertainty(0.61) imperfect(0.54) observabil(0.22) ity.(0.36) control(0.47) problem(0.67) formulated(0.87) dynamic(0.63) optimization(1.30) problem(0.58) value(0.33) function(0.28) combining(0.60) costs(0.97) rewards(0.92) multiple(0.37) steps.(0.53) paper(1.77) propose,(0.72) analyse(0.35) test(0.45) various(0.60) incremental(0.42) methods(0.43) computing(0.52) bounds(0.49) value(0.23) function(0.23) control(0.54) problems(0.62) infinite(0.38) discounted(0.33) horizon(0.62) criteria.(0.68) methods(0.60) described(0.81) tested(0.68) include(0.68) novel(1.07) incremental(0.44) versions(0.58) grid(1.00) based(0.30) linear(0.29) interpolation(0.22) method(0.39) simple(0.31) lower(0.28) bound(0.53) method(0.38) sondik(0.34) updates.(0.65) work(0.33) arbitrary(0.38) points(0.47) belief(1.86) space(0.42) enhanced(0.56) various(0.51) heuristic(0.29) point(0.31) selection(0.39) strategies.(0.64) also(0.76) introduced(1.55) new(0.88) method(0.62) computing(0.59) initial(0.46) upper(0.46) bound(0.67) fast(0.43) informed(0.56) bound(0.48) method.(0.56) method(0.44) able(0.39) improve(0.45) significantly(0.33) standard(0.37) commonly(0.37) used(0.21) upper(0.28) bound(0.40) computed(0.73) mdp(0.24) based(0.27) method.(0.41) quality(1.09) resulting(1.01) bounds(1.67) tested(1.72) maze(1.68) navigation(1.34) problem(1.05) 20(0.68) states,(0.88) 6(0.37) actions(1.42) 8(0.69) observations.(1.83) 
Node-1.4: title(0.98) learning(1.07) sorting(1.63) decision(1.56) trees(2.00) pomdps(1.04) abstract(1.34) pomdps(1.10) general(0.42) models(0.59) sequential(0.99) decisions(0.93) actions(0.63) observations(1.27) probabilistic.(0.59) many(0.37) problems(1.06) interest(0.66) formulated(0.98) pomdps,(1.14) yet(0.44) use(0.34) pomdps(0.54) limited(0.33) lack(0.32) effective(0.49) algorithms.(0.73) recently(0.51) started(0.30) change(0.25) number(0.21) problems(0.76) robot(1.39) navigation(0.71) planning(0.60) beginning(0.32) formulated(0.70) solved(0.63) pomdps.(0.59) advantage(0.46) pomdp(0.62) approach(0.80) clean(0.50) semantics(0.85) ability(0.28) produce(0.24) principled(0.33) solutions(0.42) integrate(0.41) physical(0.79) information(0.41) gathering(0.50) actions.(0.71) paper(1.89) pursue(0.91) approach(0.72) context(0.64) two(0.59) learning(0.61) tasks(0.60) learning(0.47) sort(0.46) vector(0.57) numbers(0.48) learning(0.67) decision(2.36) trees(1.73) data.(0.63) problems(1.59) formulated(1.32) pomdps(2.25) solved(0.93) general(0.57) pomdp(1.45) algorithm.(0.82) main(0.56) lessons(1.34) results(0.55) 1(0.48) use(0.39) suitable(0.36) heuristics(0.29) representations(0.49) allows(0.42) solution(0.37) sorting(1.14) classification(0.58) pomdps(0.58) non(0.22) trivial(0.45) sizes,(0.42) 2(0.29) quality(0.31) resulting(0.29) solutions(0.38) competitive(0.37) best(0.32) algorithms,(0.42) 3(0.28) problematic(0.52) aspects(0.42) decision(1.53) tree(1.12) learning(0.63) test(0.45) mis(0.29) classification(0.51) costs,(0.36) noisy(0.76) tests,(0.39) missing(0.32) values(0.36) naturally(0.66) accommodated.(0.88) 
Node-1.5: title(0.90) approximating(0.42) optimal(0.69) policies(0.65) partially(1.13) observable(0.41) stochastic(0.50) domains(0.63) abstract(1.25) problem(0.51) making(0.22) optimal(0.40) decisions(0.42) uncertain(1.01) conditions(0.80) central(0.56) artificial(0.82) intelligence.(0.75) state(0.54) world(0.72) known(0.29) times,(0.22) world(0.61) modeled(0.35) markov(0.48) decision(0.48) process(0.35) mdp(0.20) ).(0.23) mdps(0.32) studied(0.34) extensively(0.41) many(0.26) methods(0.42) known(0.30) determining(0.35) optimal(0.44) courses(0.42) action,(0.41) policies.(0.70) realistic(0.58) case(0.39) state(0.70) information(0.67) partially(0.43) observable,(0.45) partially(0.70) observable(0.31) markov(0.36) decision(0.52) processes(0.40) pomdps(0.22) ),(0.26) received(0.36) much(0.26) less(0.35) attention.(0.54) best(0.50) exact(0.48) algorithms(0.98) problems(0.75) inefficient(0.25) space(0.28) time.(0.51) introduce(2.21) smooth(0.81) partially(1.31) observable(0.52) value(0.48) approximation(0.99) spova(0.26) ),(0.35) new(1.42) approximation(1.16) method(0.80) quickly(0.52) yield(0.42) good(0.32) approximations(0.40) improve(0.33) time.(0.30) method(0.45) combined(0.42) reinforcement(0.76) learning(0.55) methods,(0.41) combination(0.44) effective(0.48) test(0.63) cases.(0.57) 
Node-1.6: title(1.58) efficient(1.08) dynamic(0.83) programming(1.15) updates(2.24) partially(0.70) observable(0.55) markov(0.87) decision(1.19) processes(0.86) abstract(1.67) examine(0.99) problem(0.72) performing(0.50) exact(0.70) dynamic(0.57) programming(0.75) updates(1.48) partially(1.26) observable(0.58) markov(0.78) decision(1.58) processes(0.78) pomdps(1.18) computational(1.04) complexity(0.75) viewpoint.(0.80) dynamic(0.72) programming(0.93) updates(1.74) crucial(0.60) operation(0.47) wide(0.25) range(0.29) pomdp(1.34) solution(0.98) methods(0.64) find(0.43) intractable(0.61) perform(0.44) updates(1.31) piecewise(0.41) linear(0.55) convex(1.90) value(0.88) functions(0.54) general(0.61) pomdps.(1.28) offer(1.36) new(1.15) algorithm,(1.71) called(1.15) witness(7.58) algorithm,(2.37) compute(0.78) updated(1.06) value(0.83) functions(0.60) efficiently(0.83) restricted(0.69) class(0.46) pomdps(0.92) number(0.57) linear(0.66) facets(0.70) great.(1.14) compare(1.60) witness(8.22) algorithm(3.33) existing(1.12) algorithms(1.68) analytically(0.87) empirically(0.88) find(0.95) fastest(1.48) algorithm(1.78) wide(0.44) range(0.44) pomdp(3.93) sizes.(1.29) 
Node-2: title(14.46) efficient(7.56) inference(7.77) bayes(5.83) networks(10.92) combinatorial(4.43) optimization(7.56) problem(7.08) abstract(20.68) number(4.43) exact(10.23) algorithms(14.38) developed(3.37) perform(4.58) probabilistic(5.11) inference(6.22) bayesian(9.36) belief(43.68) networks(17.76) recent(7.91) years.(5.88) techniques(5.10) used(3.03) algorithms(12.62) closely(3.05) related(2.74) network(7.05) structures(5.11) easy(2.52) understand(3.09) implement.(8.30) paper,(12.97) consider(9.07) problem(8.09) combinatorial(3.31) optimization(5.31) point(2.81) view(5.13) state(7.40) efficient(7.68) probabilistic(3.59) inference(7.52) belief(36.55) network(12.91) problem(5.46) finding(3.37) optimal(5.77) factoring(4.62) given(3.44) set(2.39) probability(11.09) distributions.(9.15) viewpoint,(7.21) previously(5.96) developed(4.64) algorithms(8.95) seen(6.21) alternate(6.64) factoring(3.80) strategies.(8.92) paper,(9.32) define(20.14) combinatorial(10.03) optimization(13.81) problem,(10.52) optimal(13.04) factoring(4.44) problem,(5.48) discuss(12.35) application(5.49) problem(8.90) belief(42.48) networks.(16.00) show(9.32) optimal(7.28) factoring(4.10) provides(4.40) insight(7.52) key(3.28) elements(3.51) efficient(10.67) probabilistic(5.69) inference,(7.93) demonstrate(13.60) simple,(4.61) easily(2.91) implemented(3.21) algorithms(9.27) excellent(6.64) performance.(9.49) 
Node-2.1: title(0.96) sensitivities(0.32) alternative(0.73) conditional(0.51) probabilities(0.29) bayesian(0.53) belief(1.27) networks(0.93) abstract(1.29) show(0.82) alternative(0.44) way(0.32) representing(0.60) bayesian(0.48) belief(1.94) network(1.51) sensitivities(0.25) probability(0.69) distributions.(0.55) representation(0.53) equivalent(0.34) traditional(1.06) representation(0.54) conditional(0.36) probabilities,(0.26) makes(0.24) dependencies(0.29) nodes(1.00) apparent(0.68) intuitively(0.28) easy(0.28) understand.(0.69) also(0.53) propose(1.01) qr(0.28) matrix(0.71) representation(0.65) sensitivities(0.26) conditional(0.31) probabilities(0.21) efficient,(0.25) memory(0.44) requirements(0.26) computational(0.35) speed,(0.26) traditional(0.64) representation(0.53) computer(0.52) based(0.21) implementations(0.49) probabilistic(0.21) inference.(0.62) use(0.63) sensitivities(0.51) show(1.23) certain(0.44) class(0.42) binary(0.73) networks,(0.74) computation(0.50) time(0.32) approximate(0.64) probabilistic(0.27) inference(0.62) positive(0.23) upper(0.31) bound(0.30) error(0.45) result(0.25) independent(0.32) size(0.26) network.(0.85) finally,(1.12) alternative(0.86) traditional(0.81) algorithms(1.18) use(0.44) conditional(0.79) probabilities,(0.42) describe(1.47) exact(1.02) algorithm(1.54) probabilistic(0.25) inference(0.98) uses(0.48) qr(0.27) representation(0.54) sensitivities(0.45) updates(1.16) probability(0.89) distributions(0.74) nodes(0.93) network(1.40) according(0.75) messages(1.33) neigh(1.47) bors.(0.88) 
Node-2.2: title(1.81) algebraic(1.21) techniques(0.51) efficient(0.73) inference(0.70) bayesian(0.55) networks(1.04) abstract(1.57) number(0.37) exact(0.77) algorithms(1.75) developed(0.34) perform(0.43) probabilistic(0.35) inference(0.59) bayesian(0.50) belief(1.91) networks(1.06) recent(1.20) years.(0.81) algorithms(0.79) use(0.32) graph(0.60) theoretic(0.24) techniques(0.43) analyze(0.40) exploit(0.42) network(0.50) topology.(0.79) paper,(1.07) examine(1.16) problem(0.52) efficient(0.56) probabilistic(0.25) inference(0.57) belief(1.60) network(0.88) combinatorial(0.29) optimization(1.34) problem,(0.45) finding(0.35) optimal(0.55) factoring(0.70) given(0.46) algebraic(0.78) expression(0.59) set(0.20) probability(0.69) distributions.(1.04) define(1.24) combinatorial(0.33) optimization(1.91) problem,(0.57) optimal(0.55) factoring(0.41) problem,(0.40) discuss(1.22) application(0.49) problem(0.46) belief(1.47) networks.(1.33) show(0.80) optimal(0.56) factoring(0.42) provides(0.29) insight(0.47) key(0.24) elements(0.31) efficient(0.89) probabilistic(0.32) inference,(0.53) present(0.61) simple,(0.35) easily(0.28) implemented(0.31) algorithms(0.82) excellent(0.41) performance.(0.61) also(0.64) show(1.07) use(0.41) algebraic(0.88) perspective(0.65) permits(0.76) significant(0.52) extension(0.70) belief(2.80) net(1.18) representation.(1.00) 
Node-2.3: title(1.43) interpretation(0.76) complex(0.42) scenes(0.80) using(0.46) bayesian(0.87) networks(0.89) abstract(1.12) object(0.85) recognition(1.42) systems,(0.46) interactions(0.45) objects(0.43) scene(0.56) ignored(0.38) best(0.26) interpretation(0.69) considered(0.24) set(0.22) hypothesized(0.20) objects(0.39) matches(0.23) greatest(0.26) number(0.20) image(0.50) features.(0.79) show(0.97) image(0.67) interpretation(1.10) cast(0.81) problem(0.77) finding(0.41) probable(0.63) explanation(0.60) mpe(0.22) bayesian(0.69) network(0.67) models(0.30) visual(0.64) physical(0.45) object(0.71) interactions.(0.83) problem(0.59) determine(0.39) exact(0.40) conditional(1.18) probabilities(0.62) network(1.07) shown(0.68) unimportant,(0.35) since(0.29) goal(0.42) find(0.27) probable(0.47) configuration(0.62) objects,(0.48) calculate(0.56) absolute(0.44) probabilities.(0.50) furthermore(1.36) show(0.88) evaluating(0.71) configurations(1.15) feature(1.08) counting(0.49) equivalent(0.59) calculating(0.52) joint(0.44) probability(0.57) configuration(0.93) using(0.42) restricted(0.62) bayesian(0.57) network,(0.48) derive(1.14) assumptions(0.95) probabilities(0.38) necessary(0.30) make(0.23) bayesian(0.66) formulation(0.85) reasonable.(0.80) 
Node-2.4: title(1.04) case(0.45) based(0.40) probability(0.95) factoring(0.36) bayesian(0.53) belief(1.33) networks(1.03) abstract(1.04) bayesian(0.90) network(0.97) inference(0.87) formulated(0.55) combinatorial(0.30) optimization(0.73) problem,(0.41) concerning(0.65) computation(0.55) optimal(0.54) factoring(0.40) distribution(1.54) represented(0.70) net.(2.87) since(0.28) determination(0.37) optimal(0.35) factoring(0.24) computationally(0.29) hard(0.29) problem,(0.32) heuristic(0.33) greedy(1.22) strategies(0.44) able(0.25) find(0.18) approximations(0.31) optimal(0.36) factoring(0.24) usually(0.31) adopted.(0.54) present(0.98) paper(1.04) investigate(0.86) alternative(0.66) approach(0.77) based(0.53) combination(0.44) genetic(1.07) algorithms(0.92) ga(0.80) case(0.49) based(0.74) reasoning(2.04) cbr(0.46) ).(1.16) show(0.69) use(0.44) genetic(1.01) algorithms(0.80) improve(0.37) quality(0.32) computed(0.71) factoring(0.45) case(0.37) static(0.66) strategy(0.44) used(0.25) mpe(0.28) computation(0.74) ),(0.39) combination(0.50) ga(1.10) cbr(0.60) still(0.33) provide(0.35) advantages(0.58) case(0.34) dynamic(0.64) strategies.(0.77) preliminary(1.21) results(0.84) different(0.69) kinds(0.77) nets(3.99) reported.(1.79) 
""",
        "v2": """ROOT: title experiments real time decision algorithms abstract real time decision algorithms class incremental resource bounded horvitz, 89 anytime dean, 93 algorithms evaluating influence diagrams. present test domain real time decision algorithms, results experiments several real time decision algorithms domain. results demonstrate high performance two algorithms, decision evaluation variant incremental probabilisitic inference dambrosio, 93 variant algorithm suggested goldszmidt, goldszmidt, 95 ], pk reduced. discuss implications experimental results explore broader applicability algorithms. 
Node-1: title learning policies partially observable environments scaling abstract partially observable markov decision processes pomdp model decision problems agent tries maximize reward face limited noisy sensor feedback. study pomdp motivated need address realistic problems, existing techniques finding optimal behavior appear scale well unable find satisfactory policies problems dozen states. brief review pomdp s, paper discusses several simple solution methods shows capable finding near optimal policies selection extremely small pomdp taken learning literature. contrast, show none able solve slightly larger noisier problem based robot navigation. find combination two novel approaches performs well problems suggest methods scaling even larger complicated domains. 
Node-1.1: title formal framework speedup learning problems solutions abstract speedup learning seeks improve computational efficiency problem solving experience. paper, develop formal framework learning efficient problem solving random problems solutions. apply framework two different representations learned knowledge, namely control rules macro operators, prove theorems identify sufficient conditions learning representation. proofs constructive accompanied learning algorithms. framework captures empirical explanation based speedup learning unified fashion. illustrate framework implementations two domains symbolic integration eight puzzle. work integrates many strands experimental theoretical work machine learning, including empirical learning control rules, macro operator learning, 
Node-1.2: title acting uncertainty discrete bayesian models mobile robot navigation abstract discrete bayesian models used model uncertainty mobile robot navigation, question actions chosen remains largely unexplored. paper presents optimal solution problem, formulated partially observable markov decision process. since solving optimal control policy intractable, general, goes explore variety heuristic control strategies. control strategies compared experimentally, simulation runs robot. 
Node-1.3: title incremental methods computing bounds partially observable markov decision processes abstract partially observable markov decision processes pomdps allow one model complex dynamic decision control problems include action outcome uncertainty imperfect observabil ity. control problem formulated dynamic optimization problem value function combining costs rewards multiple steps. paper propose, analyse test various incremental methods computing bounds value function control problems infinite discounted horizon criteria. methods described tested include novel incremental versions grid based linear interpolation method simple lower bound method sondik updates. work arbitrary points belief space enhanced various heuristic point selection strategies. also introduced new method computing initial upper bound fast informed bound method. method able improve significantly standard commonly used upper bound computed mdp based method. quality resulting bounds tested maze navigation problem 20 states, 6 actions 8 observations. 
Node-1.4: title learning sorting decision trees pomdps abstract pomdps general models sequential decisions actions observations probabilistic. many problems interest formulated pomdps, yet use pomdps limited lack effective algorithms. recently started change number problems robot navigation planning beginning formulated solved pomdps. advantage pomdp approach clean semantics ability produce principled solutions integrate physical information gathering actions. paper pursue approach context two learning tasks learning sort vector numbers learning decision trees data. problems formulated pomdps solved general pomdp algorithm. main lessons results 1 use suitable heuristics representations allows solution sorting classification pomdps non trivial sizes, 2 quality resulting solutions competitive best algorithms, 3 problematic aspects decision tree learning test mis classification costs, noisy tests, missing values naturally accommodated. 
Node-1.5: title approximating optimal policies partially observable stochastic domains abstract problem making optimal decisions uncertain conditions central artificial intelligence. state world known times, world modeled markov decision process mdp ). mdps studied extensively many methods known determining optimal courses action, policies. realistic case state information partially observable, partially observable markov decision processes pomdps ), received much less attention. best exact algorithms problems inefficient space time. introduce smooth partially observable value approximation spova ), new approximation method quickly yield good approximations improve time. method combined reinforcement learning methods, combination effective test cases. 
Node-1.6: title efficient dynamic programming updates partially observable markov decision processes abstract examine problem performing exact dynamic programming updates partially observable markov decision processes pomdps computational complexity viewpoint. dynamic programming updates crucial operation wide range pomdp solution methods find intractable perform updates piecewise linear convex value functions general pomdps. offer new algorithm, called witness algorithm, compute updated value functions efficiently restricted class pomdps number linear facets great. compare witness algorithm existing algorithms analytically empirically find fastest algorithm wide range pomdp sizes. 
Node-2: title efficient inference bayes networks combinatorial optimization problem abstract number exact algorithms developed perform probabilistic inference bayesian belief networks recent years. techniques used algorithms closely related network structures easy understand implement. paper, consider problem combinatorial optimization point view state efficient probabilistic inference belief network problem finding optimal factoring given set probability distributions. viewpoint, previously developed algorithms seen alternate factoring strategies. paper, define combinatorial optimization problem, optimal factoring problem, discuss application problem belief networks. show optimal factoring provides insight key elements efficient probabilistic inference, demonstrate simple, easily implemented algorithms excellent performance. 
Node-2.1: title sensitivities alternative conditional probabilities bayesian belief networks abstract show alternative way representing bayesian belief network sensitivities probability distributions. representation equivalent traditional representation conditional probabilities, makes dependencies nodes apparent intuitively easy understand. also propose qr matrix representation sensitivities conditional probabilities efficient, memory requirements computational speed, traditional representation computer based implementations probabilistic inference. use sensitivities show certain class binary networks, computation time approximate probabilistic inference positive upper bound error result independent size network. finally, alternative traditional algorithms use conditional probabilities, describe exact algorithm probabilistic inference uses qr representation sensitivities updates probability distributions nodes network according messages neigh bors. 
Node-2.2: title algebraic techniques efficient inference bayesian networks abstract number exact algorithms developed perform probabilistic inference bayesian belief networks recent years. algorithms use graph theoretic techniques analyze exploit network topology. paper, examine problem efficient probabilistic inference belief network combinatorial optimization problem, finding optimal factoring given algebraic expression set probability distributions. define combinatorial optimization problem, optimal factoring problem, discuss application problem belief networks. show optimal factoring provides insight key elements efficient probabilistic inference, present simple, easily implemented algorithms excellent performance. also show use algebraic perspective permits significant extension belief net representation. 
Node-2.3: title interpretation complex scenes using bayesian networks abstract object recognition systems, interactions objects scene ignored best interpretation considered set hypothesized objects matches greatest number image features. show image interpretation cast problem finding probable explanation mpe bayesian network models visual physical object interactions. problem determine exact conditional probabilities network shown unimportant, since goal find probable configuration objects, calculate absolute probabilities. furthermore show evaluating configurations feature counting equivalent calculating joint probability configuration using restricted bayesian network, derive assumptions probabilities necessary make bayesian formulation reasonable. 
Node-2.4: title case based probability factoring bayesian belief networks abstract bayesian network inference formulated combinatorial optimization problem, concerning computation optimal factoring distribution represented net. since determination optimal factoring computationally hard problem, heuristic greedy strategies able find approximations optimal factoring usually adopted. present paper investigate alternative approach based combination genetic algorithms ga case based reasoning cbr ). show use genetic algorithms improve quality computed factoring case static strategy used mpe computation ), combination ga cbr still provide advantages case dynamic strategies. preliminary results different kinds nets reported. 
""",
    }

    @classmethod
    def get(cls, version: Union[Literal["latest"], str], style: Literal["json", "document"]):
        if style == "json":
            examples = cls.example_jsonified_graph
        elif style == "document":
            examples = cls.example_graph

        if version == "latest":
            return list(examples.values())[-1]

        return examples[version]


class FreeTextExpGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.example = None
        self.template = None

    @staticmethod
    def save(prompt, response, output_file, system_prompt="You are a helpful AI assistant.", style="OpenAI"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                },
                f,
            )

    def gen(
        self,
        label,
        document,
        model=None,
        template_version="latest",
        example_version="latest",
        example_style="document",
        use_finetuned=False,
    ):
        self.template = TemplatesArchive.get(version=template_version)
        self.example = ExampleGraphCorpus.get(version=example_version, style=example_style)

        prompt = self.template.format(label=label, document=document, example_verbalized_graph=self.example)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful, reliable and responsible AI assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        if not use_finetuned:
            assert model is not None, "model is required when not using finetuned model"
            # Call OpenAI GPT API
            completion = self.client.chat.completions.create(
                model=model,
                temperature=0.3,
                top_p=0.95,
                max_tokens=4095,
                messages=messages,
            )
            response = completion.choices[0].message.content
            return prompt, response

        else:
            # Use finetuned distilled model
            from distill import LLMFinetuner

            finetuner = LLMFinetuner(model_name=model)
            response = finetuner.inference(messages)
            return prompt, response


def generate_explanations(indexs, model_id: str, exp_id: Union[int, str]):
    generator = FreeTextExpGenerator()

    # input dir
    pkl_dir = Path("outputs/pkls/")
    # output dir
    exp_dir = Path(f"outputs/generated_expls-{exp_id}")
    exp_dir.mkdir(exist_ok=True)

    for index in indexs:
        output_file = exp_dir / f"{index}.json"
        if output_file.exists():
            continue

        try:
            graph = TAG.load(pkl_dir / f"{index}.pkl")
        except FileNotFoundError:
            logger.error(pkl_dir / f"{index}.pkl does not exists!")
            continue
        except Exception as exc:
            logger.error("Error: %s", exc)
            exit(1)

        start = time.time()
        label = graph.prediction
        try:
            document = graph.text(style="document", masker="", with_score=True)
        except Exception as e:
            logger.error("stringify graph-%d failed! Error: %s", index, e)
            continue

        prompt, response = generator.gen(
            model=model_id,
            label=label,
            document=document,
        )

        FreeTextExpGenerator.save(prompt, response, output_file)

        end = time.time()
        logger.info("explanation-%d generated. Time: %.2fs", index, end - start)
