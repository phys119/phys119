# Review of Lab 01 uncertainty concepts

## Introduction

Let's review what we learned about characterizing uncertainties in Lab 01. During Lab 01 (the full set of slides, with answers, has been posted), we discussed how to estimate our confidence in a measurement and turn that into a standard uncertainty, which we represented as δ(..). We discussed three different situations so far.

1. The measurement uses a digital instrument where the main limitation is the precision (rounding) of the number displayed,
2. The measurement is fluctuating due to some source of noise, and
3. The measurement uses an analog scale where there is a limitation on how well you can judge the measured value.

In Lab 01, we also introduced the idea of Rectangular and Gaussian PDFs (Probability Density Functions). In the next few questions, we are going to use our knowledge of these two PDFs to calculate the uncertainty associated with the four measurements that we did in Lab 01, which were

1. Uncompressed length of spring (x1),
2. Compressed length of spring (x2),
3. Uncompressed mass of spring (m1), and
4. Compressed mass of spring (m2).

Remember that the standard uncertainty "δ(..)" for a single measurement is the standard deviation "σ" in our distribution. The standard deviation of each PDF can be estimated using the formulae below.

$$\sigma_{gaussian} = \frac{95\% \text{ Confidence Interval (CI)}}{4}$$

$$\sigma_{rectangular} = \frac{a \text{ (half-width)}}{\sqrt{3}}$$

---

## Questions

### Question 1

Which probability density function (PDF) should you use to determine the standard uncertainty of a single measurement from a digital output?

<ol type="A">
<li>Rectangular PDF</li>
<li>Gaussian PDF</li>
<li>It depends on whether the reading from the digital output is stable or not</li>
</ol>

### Question 2

Which probability density function (PDF) should you use to determine the standard uncertainty associated with the uncompressed *length* of spring measurement?

<ol type="A">
<li>Gaussian PDF</li>
<li>Rectangular PDF</li>
<li>Both PDFs</li>
</ol>

### Question 3

Consider a measurement of spring position x2 at 5 cm. If you decide that the 95% confidence interval spans a range from 4.9 to 5.1 cm, what should you report as your standard uncertainty in the measurement u[x2]?

<ol type="A">
<li>u[x2] is 0.050 cm</li>
<li>u[x2] is 0.10 cm</li>
<li>u[x2] is 0.20 cm</li>
</ol>

### Question 4

What is the uncertainty associated with the uncompressed mass of the spring measurement (m1) given that the last digit (and smallest possible change in the reading) of the measuring balance is 1 gram?

<ol type="A">
<li>1.0 g</li>
<li>0.50 g</li>
<li>0.29 g</li>
</ol>

### Question 5

How do we calculate the standard uncertainty associated with the compressed mass of spring measurement (m2), given that we see the digital number fluctuating?

<ol type="A">
<li>Take the smallest scale of the measuring balance</li>
<li>Take the smallest scale of the measuring balance and divide by two</li>
<li>Calculate the standard deviation of the Gaussian PDF using (95% CI)/4</li>
<li>Calculate the standard deviation of the Rectangular PDF using half-width/sqrt(3)</li>
</ol>

---

## Answers

### Question 1

**Answer: C. It depends on whether the reading from the digital output is stable or not**

If the reading from the digital output is stable, the uncertainty associated with it will be represented by a Rectangular PDF since the instrumental uncertainty dominates. However, if the reading is unstable, the uncertainty is dominated by the random uncertainty, in which case the standard uncertainty should be represented by a Gaussian PDF.

### Question 2

**Answer: A. Gaussian PDF**

As experimenters we need to make many small judgements in order to do this measurement (e.g., Where is the top of the spring? How does that line up on the ruler sitting a couple of cm away? How is is the ruler to read?). You can still determine a most likely answer, so the appropriate distribution is the peaked one, not a flat rectangular one. Therefore, the standard uncertainty of the measurement should be estimated using a Gaussian PDF.

### Question 3

**Answer: A. δx<sub>2</sub> is 0.050 cm**

We know that the standard uncertainty associated with the length of spring measurement should be estimated using a Gaussian distribution (from Question 2). Hence, we can calculate the standard uncertainty from the relationship that standard uncertainty of a Gaussian is one quarter of the 95% CI:

*u[x2] = (95% CI)/4*

*u[x2] = (5.1 cm − 4.9 cm)/4*

*u[x2] = 0.050 cm*

Note that we express this as 0.050 cm and not 0.05 cm because of the convention that we report standard uncertainties to 2 significant figures.

### Question 4

**Answer: C. 0.29 g**

We observed that the spring in its uncompressed state gave a stable digital reading (i.e. the value displayed was not fluctuating). Hence, the only uncertainty associated with the measurement is the instrumental uncertainty. We represented that measurement as a rectangular PDF that was 1 gram wide (so the half-width is a = 0.5 grams). The standard uncertainty is the standard deviation of this rectangular distribution, a/sqrt(3) = 0.5 g/sqrt(3) = 0.29 g.

Note that we keep 2 significant figures when we report a standard uncertainty.

### Question 5

**Answer: C. Calculate the standard deviation of the Gaussian PDF using (95% CI)/4**

Back in the first question we concluded that the type of PDF that we should use to determine the standard uncertainty of a measurement from a digital output depends on whether the reading from the digital output is stable or not. We observed that values displayed on the mass balance when we were compressing the spring were not steady, and hence we could represent the standard uncertainty of that measurement using a Gaussian PDF. We decided it would be reasonable to estimate a 95% CI for that measurement by observing the maximum (m_max) and minimum (m_min) values displayed when we tried to hold the force constant. If we treat the 95% CI as "Max−Min", the standard uncertainty would be:

*u[m2] = (m_max − m_min)/4*
