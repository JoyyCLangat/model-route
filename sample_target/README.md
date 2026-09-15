# sample_target

A tiny, deliberately buggy Python project. It exists so the demo has
something real to work on, and so you do not need a repo of your own handy to
try the router.

The bugs are ordinary ones: a mutable default argument, an off by one loop, a
discount that subtracts dollars instead of a percentage, a mean that divides
by zero on an empty list, and a clamp that returns the wrong bound. The point
of the demo is to watch the router send the reading work to a large context
model and the patch work to a capable one, not to be impressed by the fixes.
