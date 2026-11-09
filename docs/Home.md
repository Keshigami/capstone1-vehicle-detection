# Vehicle detection and Tesla autopilot safety

A Keras model that names the main vehicle in a traffic-camera frame and boxes it, plus the data analysis of Tesla's fatal-crash sheet. Caltech CTME PGC AIML, capstone, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/capstone1-vehicle-detection/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
Intelligent transport systems need to count, track and classify vehicles from roadside cameras in
real time. Part 1 asks for a deep-learning model that says what kind of vehicle is in a frame
(car, bus, pickup, truck, bicycle and so on) and where it is, as a rectangle. Part 2 asks what the
public "Tesla Deaths" sheet says about Autopilot and road safety: events by year, country and
model, who died, and how many Autopilot deaths are claimed versus verified. Solved means a page
where you drop a camera frame and get a labelled box, and a page that answers the safety questions
with numbers that can be checked.

## The one-line answer
73.0% · val accuracy · 900 held-out frames · baseline 58.7%

## One honest limit
One box per frame. IoU 0.25: boxes are loose.
