# Chapter 1 - Motivation for Explainable AI

Modern machine learning systems can classify images, recommend products, detect fraud, and support medical decisions with impressive accuracy. Yet high predictive performance does not automatically mean that a system is safe, fair, reliable, or worthy of human trust. In many real-world settings, the main problem is not that a model produces answers, but that it produces them in ways that humans cannot properly inspect, question, or contest. This is the central motivation for explainable artificial intelligence (XAI).

The rise of XAI is therefore not just a technical trend. It is a response to a broader failure in the way machine learning systems are often evaluated. Accuracy, loss, and benchmark performance matter, but they do not capture everything humans care about. When an AI system is used in hiring, medicine, criminal justice, public administration, or scientific research, people also want to know whether the model is using sensible evidence, whether it is relying on hidden shortcuts, whether it can be challenged, and whether its decisions can be justified to those affected by them. This chapter introduces the main ideas behind that demand. It explains what the "black box" problem is, why explainability becomes necessary, who needs explanations, what makes an explanation useful, and why interpretability is tied to trust, accountability, and the historical development of AI itself.

## 1. The Black Box Problem

The phrase *black box* usually refers to a system whose inputs and outputs are visible, but whose internal reasoning is difficult for humans to understand. In machine learning, this label is often applied to models such as deep neural networks, ensembles, or any system whose decision process is too complex to inspect directly. However, the black box problem is not simply about model complexity. A model becomes a black box when people cannot meaningfully determine why it produced a decision, what evidence mattered most, or whether its reasoning matches the real task.

This distinction matters because a model can appear successful while still learning the wrong lesson. A useful analogy comes from the famous story of Clever Hans, a horse that appeared to solve arithmetic problems in the early twentieth century. Hans was not actually doing arithmetic. He was responding to subtle cues from the humans around him. From the outside, the behavior looked intelligent; closer inspection showed that the apparent success depended on an unintended shortcut. Machine learning models can behave in a similar way. A classifier may achieve excellent performance by relying on patterns that happen to work in one dataset but fail in the real world.

Interpretability papers and teaching material often use a simple wolf-versus-husky illustration to explain this idea. In that example, a classifier appears to separate wolves from huskies, but the explanation suggests that it is relying mainly on snow in the background rather than on the animal itself. The example is mainly used to show how explanations can reveal a shortcut that would remain hidden if one looked only at the prediction result (Ribeiro et al., 2016; Molnar, 2024).

A paper-backed medical example makes the same point more clearly. Zech et al. studied deep learning models for pneumonia detection from chest X-rays collected at different hospitals. The problem was that each hospital did not only differ in patients; it also differed in image quality, equipment, markings, and disease prevalence. As a result, a model could partly learn to identify the hospital that produced the image and then use that information as a shortcut. If one hospital had more pneumonia cases than another, hospital identity itself became predictive. This helped the model inside the original dataset, but it hurt performance when the model was tested on images from a different hospital system (Zech et al., 2018).

The black box problem, then, is not merely that a model is mathematically complicated. The deeper problem is that human observers may be unable to tell whether the model is making decisions for the right reasons. That is precisely why explanation becomes important.

## 2. Interpretability, Explainability, and Incompleteness

Although the terms *interpretability* and *explainability* are often used interchangeably, the literature does not treat them as perfectly identical. A common starting point is the idea that interpretability concerns whether a human can understand the cause of a decision, while explainability concerns the ability of a system or method to provide an understandable account of that decision. In practice, both ideas are human-centered: they matter because some person must be able to make sense of the model's behavior.

Doshi-Velez and Kim define interpretability as the ability to explain or present something in understandable terms to a human. This is intentionally broad, because what counts as understandable depends on the person, the task, and the stakes (Doshi-Velez and Kim, 2017). Lipton similarly argues that "interpretability" is not a single property. Sometimes people mean transparency of the model itself. In other cases they mean a post-hoc explanation that is produced after the model has already been trained. Treating all of these goals as one thing creates confusion and makes the field less rigorous (Lipton, 2018).

One of the most important ideas in the motivation for XAI is *incompleteness*. Doshi-Velez and Kim argue that explanations are needed when the formal problem definition does not fully capture what humans actually care about. A model can optimize an objective perfectly and still fail at the broader task because the metric is incomplete (Doshi-Velez and Kim, 2017). This is different from uncertainty. Uncertainty can often be measured: for example, a classifier might output a probability or confidence score. Incompleteness is harder. It arises when important values, constraints, or background assumptions cannot be fully written into the objective function.

That is why accuracy alone is often insufficient. In a medical setting, a model may have strong benchmark performance, but doctors still need to know whether it relies on clinically meaningful cues. In criminal justice, a risk score may be statistically predictive, but affected individuals and regulators may still ask whether the reasoning is fair, contestable, and legally defensible. In scientific discovery, researchers may care less about raw prediction than about whether the model reveals a plausible mechanism. In each case, explanation is needed because the formal target captures only part of the real task.

Molnar's overview of interpretable machine learning is useful here because it puts the idea in practical terms. A model can be good at prediction and still be hard to use responsibly. People often need more than an output score. They need to check for mistakes, explain decisions to others, learn something from the model, or decide what action to take next (Molnar, 2024). In that sense, XAI becomes necessary when the training objective is not enough for the real human task.

## 3. Why Explainability Matters

There is no single reason to want explanations. Different domains demand them for different purposes. Still, several motivations appear repeatedly across the literature.

### 3.1 Safety and reliability

In high-stakes systems, incorrect reasoning can be more dangerous than low average accuracy suggests. A model that works most of the time but fails for hidden reasons may be unsafe in deployment. Explanations help users inspect whether the system is relying on robust evidence or on brittle shortcuts that will collapse when the environment changes. This is particularly important in medicine, autonomous systems, and critical infrastructure.

### 3.2 Debugging and model improvement

Interpretability is often a debugging tool. By examining which features, examples, or internal patterns matter most, practitioners can discover data leakage, misleading correlations, poor feature engineering, label problems, or biased proxy variables. In this sense, explanation improves model development. It does not only justify a finished system; it helps identify why the system should not yet be trusted. A well-known healthcare example is the pneumonia-risk model discussed by Caruana et al. Their interpretable analysis revealed that a history of asthma appeared to lower the predicted risk of death from pneumonia. Medically, that conclusion was suspicious. The pattern made sense only after clinicians realized that asthma patients were being treated more aggressively and admitted earlier, which changed outcomes in the data. Without interpretation, a model could have turned that pattern into a dangerous rule (Caruana et al., 2015; Molnar, 2024).

### 3.3 Scientific discovery

In some applications, the prediction is only the beginning. Researchers may use machine learning to study biology, medicine, agriculture, or social behavior because they want to learn something about the world, not only predict the next label. In these settings, a highly accurate but opaque model is limited. It may tell a scientist *what* is likely to happen without showing *why* the model believes that. Interpretability can make those learned patterns more useful. For example, a model that predicts crop yield is much more valuable if it also helps researchers examine how fertilizer, weather, or soil conditions affect the prediction. In that setting, the model becomes a tool for generating scientific insight, not just a black box that produces numbers (Molnar, 2024).

### 3.4 Human trust and decision support

Many machine learning systems are used to support, rather than replace, human decision-makers. A doctor, manager, loan officer, or policy analyst may be expected to act on the model's output. But acting responsibly requires some basis for judgment. Users often want to know why the model reached its recommendation, whether similar cases were treated in the same way, and how confident the system is under distribution shift or unusual inputs.

### 3.5 Recourse for affected people

People who are directly affected by automated decisions often need explanations for practical reasons. A rejected loan applicant, denied insurance customer, or flagged job candidate may want to know what contributed to the decision and what could be changed in the future. Explanation, in this context, is not merely informative. It is tied to recourse, contestability, and procedural fairness.

Taken together, these motivations show why XAI should not be reduced to a vague desire for "transparency." Explanation matters because machine learning systems are embedded in human institutions where performance metrics are incomplete proxies for real goals.

## 4. What Makes a Good Explanation?

If explanations are important, the next question is what counts as a good one. Research in psychology and XAI suggests that useful explanations are not simply long descriptions of everything the model computed. Human beings do not usually want exhaustive causal inventories. They want explanations that help them answer a relevant question.

One important property is *contrastiveness*. People often ask not "Why did this happen?" but "Why did this happen instead of something else?" A loan applicant may want to know why the application was rejected instead of approved. A doctor may want to know why the system predicts pneumonia rather than heart failure. Contrastive explanations are especially valuable because they support action: they show what matters relative to an alternative.

Another property is *selectivity*. Humans can process only a limited number of reasons at once. A useful explanation therefore highlights the most important factors instead of presenting an unstructured list of all possible contributors. Selectivity does not mean dishonesty. It means that explanations should be cognitively manageable.

Explanations are also *social*. They are given to someone, in a context, for a purpose. The same model decision may need to be explained differently to a regulator, a patient, a data scientist, and a business executive. An explanation that is mathematically precise but inaccessible to its audience may still fail as an explanation.

Miller's work on explanation in AI also emphasizes that people are often more interested in unusual, abnormal, or decision-changing factors than in background conditions that are always present (Miller, 2019). That is why saying that a house is expensive because it has a roof is not very helpful, while saying that it is expensive because it has a rare second balcony may be informative. Good explanations focus attention on what is meaningful in context.

At the same time, explanation quality cannot be judged only by how persuasive it sounds. A smooth explanation may be psychologically satisfying while misrepresenting the real model. This is one of the major risks of post-hoc XAI methods: they can produce plausible stories that humans like, even when those stories are not faithful to the true decision process. A good explanation must therefore balance human usefulness with fidelity to the underlying model behavior.

## 5. Who Needs Explanations?

One reason explainability is difficult is that different stakeholders need different kinds of understanding. A single explanation style rarely satisfies everyone.

Creators, such as data scientists and engineers, often need explanations for debugging, validation, and model improvement. They want to know whether the system has learned robust structure or whether it is exploiting shortcuts in the training data.

Operators interact directly with the system in practice. They may need concise, case-specific explanations that help them decide whether to rely on the output in a particular moment.

Executors make decisions based on the AI's output. For example, a hiring manager or clinician may need enough understanding to justify an action and to know when human judgment should override the model.

Decision subjects are people directly affected by automated decisions. They may care less about the model's architecture than about whether the outcome was fair, whether it can be challenged, and what changes could alter future outcomes.

Auditors and regulators need explanations at the system level. Their concern is often whether a model is compliant, traceable, and consistent with legal or institutional standards.

Data subjects form another important group. These are people whose data helped train the system, even if they are not the immediate targets of a given decision. They may have concerns about how their data were used and whether the resulting model reproduces harmful social patterns.

The stakeholder perspective shows why explainability is not a single interface problem. It is a problem of matching forms of understanding to different roles, rights, and responsibilities.

## 6. Trust, Accountability, and Contestability

Interpretability is often linked to trust, but that relationship is more complicated than it first appears. It is tempting to assume that a more interpretable model is automatically more trustworthy. Lipton warns against this assumption. Sometimes an explanation reveals that the model is using a biased proxy, a shortcut that is not actually related to the task, or a pattern that is too weak to rely on. In such cases, interpretation should reduce trust, not increase it.

Trust therefore needs to be understood more carefully. A person may want confidence in at least three things: that the model performs well outside the lab, that it is not systematically unfair, and that its mechanism is understandable enough to justify reliance. These are related but distinct concerns. A model can score well on one and poorly on the others (Lipton, 2018).

This is why accountability matters. The FAT/ML principles for accountable algorithms argue that there is always a human ultimately responsible for algorithmic systems and their consequences. "The algorithm did it" is not an acceptable excuse. Responsibility, explainability, accuracy, auditability, and fairness all become part of a broader governance problem: if an automated system causes harm, who can inspect it, question it, and change it? (FAT/ML, 2016).

From the perspective of affected individuals, accountability is closely tied to *contestability*. A system is not meaningfully accountable if people cannot challenge its outcomes. That challenge requires more than access to source code. In many cases, affected individuals need an understandable explanation, a channel for recourse, and access to human review with real authority. Otherwise, explanation becomes symbolic rather than practical.

This issue appears in legal discussions as well. Goodman and Flaxman's analysis of the GDPR popularized the debate over a "right to explanation," but the more cautious and precise point is that EU data protection law gives individuals rights against certain forms of solely automated decision-making and includes rights such as obtaining human intervention, expressing a point of view, and contesting the decision under Article 22 (Goodman and Flaxman, 2017; GDPR, 2016). More recently, the EU AI Act reinforced the broader regulatory shift by treating transparency, documentation, traceability, and human oversight as central governance requirements for many higher-risk AI deployments (AI Act, 2024). The important idea for XAI is that explanation is not only about understanding a model. It is also about process: who can question a decision, who can review it, and who is responsible for it.

## 7. Case Studies: When Black Boxes Fail

Case studies make the need for explainability more concrete because they show how black box behavior can fail in practice.

### 7.1 Shortcut learning and the wolf-versus-husky example

The wolf-versus-husky example appears frequently in discussions of interpretability. A classifier seems to distinguish wolves from huskies, but the explanation indicates that it is focusing on snow or a bright background rather than on the animal itself. This shows a central problem in explainable AI: a model can appear successful while solving an easier and less meaningful problem than the one humans intended (Ribeiro et al., 2016; Molnar, 2024).

Geirhos et al. give this pattern a broader name: *shortcut learning*. Their argument is that deep models often rely on decision rules that are easier for the model to use, even if those rules are not the ones humans care about. A model may therefore score well on familiar data while failing when the background changes, the population changes, or the task moves into a new environment (Geirhos et al., 2020). Interpretability is valuable here because it helps reveal whether the model has learned the intended signal or only a convenient shortcut.

### 7.2 Hospital generalization failure in medical imaging

Zech et al. provide a stronger real-world example. They trained and tested pneumonia detection models using chest X-rays from multiple hospital systems. At first glance, the results looked promising. However, hospitals do not produce identical data. They may use different scanners, different image processing pipelines, different text markers on images, and they may treat patient populations with different pneumonia rates. Because of that, a model can quietly learn to answer a simpler question such as "Which hospital produced this image?" and then use hospital identity as a clue for predicting pneumonia.

This matters because a shortcut like hospital identity may work well inside the original dataset. If Hospital A has more pneumonia cases than Hospital B, then knowing the hospital already gives the model useful information. But that is not the same as learning the visual signs of pneumonia itself. When the model is moved to a different hospital system, that shortcut becomes much less useful, and performance can drop sharply. This is why the Zech et al. study is so important for XAI: it shows that a model can look medically strong while still relying partly on the wrong signal (Zech et al., 2018).

### 7.3 COMPAS and contestability

Another influential example comes from ProPublica's investigation of the COMPAS risk assessment tool used in U.S. criminal justice settings. The case became central to debates about algorithmic fairness because it raised questions about racial disparities, hidden proxies, and the difficulty of contesting proprietary decision systems (ProPublica, 2016). Whatever position one takes in the statistical debate around COMPAS, the broader XAI lesson is clear: in high-stakes settings, opaque systems create serious problems for accountability, trust, and public legitimacy.

These cases support the same conclusion from different angles. Explainability is needed not because humans are curious, but because models can fail in ways that remain invisible without structured inspection.

## 8. Evaluating Interpretability

If interpretability matters so much, how should it be evaluated? Doshi-Velez and Kim argue that the field often makes claims about interpretability without matching those claims to appropriate evidence. Their taxonomy remains one of the clearest frameworks for making evaluation more rigorous (Doshi-Velez and Kim, 2017).

The first category is *application-grounded evaluation*. This is the strongest and most realistic form of evaluation because it studies real users doing their real task. Suppose a system is meant to help radiologists inspect chest X-rays. An application-grounded study would ask radiologists to use the model and its explanations in a setting close to actual clinical work, then measure whether the explanations help them detect errors, make better decisions, or reject unsafe predictions. The advantage is realism. The difficulty is cost: such studies require experts, careful design, and domain-specific evaluation.

The second category is *human-grounded evaluation*. This still involves people, but the task is simplified. Instead of testing the explanation inside a hospital or courtroom, researchers might ask participants to use an explanation to predict what the model will do next, compare two explanations and choose which one is clearer, or identify what input change would reverse a prediction. These tasks do not fully reproduce real deployment, but they are useful because they test whether a human can actually understand and work with the explanation.

The third category is *functionally-grounded evaluation*. This uses no human subjects at all. Instead, it evaluates properties that researchers believe are related to interpretability, such as using fewer rules, fewer features, a shallower tree, or a monotonic relationship between input and output. For example, a sparse rule list may be treated as more interpretable than a highly tangled model. This kind of evaluation is fast and convenient, especially in early method development, but it is also limited. A mathematically simple model is not automatically easy for a person to understand in practice.

The key lesson is that evaluation should match the claim. If a paper claims that a method helps doctors make safer decisions, then a purely mathematical proxy is not enough. If a method claims only to produce a simpler model, proxy-based evaluation may be acceptable. The broader scientific point is that XAI should not depend on vague assertions that a method is "intuitive." It should specify for whom, for what task, and according to what evidence.

## 9. Historical Context: From Intrinsic Interpretability to XAI

The demand for explainability may feel contemporary, but its roots are much older. In classical statistics, interpretability was often built into the modeling approach. Linear models were valued not only for prediction but also because coefficients could be tied to meaningful variables and effects. The goal was not just to output a number, but to represent a relationship in understandable terms.

In the expert systems era of the 1970s and 1980s, explanation was also central. Systems such as MYCIN used rule-based reasoning in medicine and were designed so that their recommendation paths could be shown step by step (Buchanan and Shortliffe, 1984). The model did not need a separate explanation module because its structure itself was meant to be understandable.

The rise of modern machine learning changed that balance. As predictive performance improved through complex statistical methods, ensemble models, and deep neural networks, the field shifted away from intrinsically transparent reasoning and toward accuracy-oriented optimization. This brought enormous practical success, but it also created systems whose internal logic was difficult to interpret.

XAI can be understood as the modern response to that shift. In some cases, the response is to prefer simpler, intrinsically interpretable models where possible. In other cases, the response is post-hoc explanation: methods such as LIME, SHAP, saliency maps, feature importance scores, concept-based explanations, or example-based explanations that try to make a complex model more inspectable after training. Lipton's critique is important here. The historical story is not simply "linear models are interpretable and neural networks are not." A huge linear model with opaque engineered features may also be difficult to understand (Lipton, 2018; Molnar, 2024). The real question is what kind of understanding is needed and whether the chosen method provides it.

## 10. From Motivation to Methods

This chapter has focused on why explainability is needed, not on the full toolbox of methods used to achieve it. That distinction matters. Before choosing a technique, one must first understand the purpose of explanation. Are we trying to debug a model, justify a decision, support recourse, discover scientific structure, or satisfy regulatory requirements? Different goals require different forms of explanation, and later chapters can address those methods in greater detail.

## 11. Conclusion

Explainable AI begins from a simple observation: good predictions are not enough. In many real-world tasks, the formal objective captures only part of what humans care about. Safety, fairness, scientific understanding, contestability, and institutional accountability often remain outside the metric. That gap is what makes explanation necessary.

The motivation for XAI is therefore broader than a demand for transparency in the abstract. It is a response to incompleteness, hidden shortcuts, and the need for models that can be inspected, challenged, and responsibly integrated into human decision-making. Black box systems are not problematic merely because they are complex. They are problematic when their complexity prevents people from determining whether the system is reasoning well, failing dangerously, or exercising power without adequate justification.

As machine learning moves deeper into socially significant domains, explainability becomes part of how we align technical systems with human goals. It helps improve models, supports users, protects affected individuals, and strengthens the conditions under which automated decisions can be trusted. For those reasons, XAI is not a peripheral concern in modern AI. It is one of the field's central responses to the limits of prediction alone.

## References

1. Biran, O., and Cotton, C. (2017). *Explanation and justification in machine learning: A survey*. Proceedings of the IJCAI 2017 Workshop on Explainable AI.
2. Buchanan, B. G., and Shortliffe, E. H. (1984). *Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project*. Addison-Wesley. Chapter 11 available at: https://people.dbmi.columbia.edu/~ehs7001/Buchanan-Shortliffe-1984/Chapter-11.pdf
3. Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., and Elhadad, N. (2015). *Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission*. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1721-1730. https://doi.org/10.1145/2783258.2788613
4. Doshi-Velez, F., and Kim, B. (2017). *Towards a rigorous science of interpretable machine learning*. arXiv. https://arxiv.org/abs/1702.08608
5. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence, 2, 665-673.
6. Goodman, B., and Flaxman, S. (2017). *European Union regulations on algorithmic decision-making and a "right to explanation"*. AI Magazine, 38(3), 50-57. Preprint: https://arxiv.org/abs/1606.08813
7. Kim, B., Khanna, R., and Koyejo, O. O. (2016). *Examples are not enough, learn to criticize! Criticism for interpretability*. NeurIPS.
8. Lipton, Z. C. (2018). *The mythos of model interpretability*. Communications of the ACM, 61(10), 36-43. Preprint: https://arxiv.org/abs/1606.03490
9. Miller, T. (2019). *Explanation in artificial intelligence: Insights from the social sciences*. Artificial Intelligence, 267, 1-38.
10. Molnar, C. (2024). *Interpretable Machine Learning* (3rd ed.). https://christophm.github.io/interpretable-ml-book/
11. Partnership on AI / FAT/ML. (2016). *Principles for accountable algorithms and a social impact statement for algorithms*. https://www.fatml.org/resources/principles-for-accountable-algorithms
12. ProPublica. (2016). *Machine bias: There is software used across the country to predict future criminals. And it is biased against Blacks.* https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
13. Regulation (EU) 2016/679 of the European Parliament and of the Council (General Data Protection Regulation). Official text: https://eur-lex.europa.eu/eli/reg/2016/679/oj
14. Regulation (EU) 2024/1689 of the European Parliament and of the Council (Artificial Intelligence Act). Official text: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
15. Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144. Preprint: https://arxiv.org/abs/1602.04938
16. Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., and Oermann, E. K. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study*. PLOS Medicine, 15(11), e1002683. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
