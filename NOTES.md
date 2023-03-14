## TAL file format.

```
<name>
<total training iters>
<reward recording freq>
[<...recorded reward>]
<agent data>
```

## Env Notes.

Say X orders are set to come in.

 - Reward of 100 on completion of all orders.
 - Reward of 10 on completion of 1 order.
 - Reward of 1 on picking up a needed resource.

For orders coming in, a probability distribution is needed.

 - A, for order arrival. Binomial?
 - C, for order composition. Just a single one? What about one for each depot?
   - Or just gaussian over a random order of depots? I.e. 
   - 1 2 5 5 2 1 <- pretend-y gaussian
   - B A E F D C <- depot

# Notes from WK7

Come up with a good baseline and bad baseline. Must be well known.
Look at other work and justify the reward structure, discuss in report as it's not clear and offer some args.
perhaps could encode order queue with some features extracted.
Start getting scenarios for eval of increasing complexity.


# Notes for WK8

 - HDQN trains slowly because the upper-level network takes much fewer steps.
   - Pretrain the controller?
 - Read papers that tackle the same problem, pretty basic state spaces :(
   - Some don't even consider items within an order.
   - Some give orders priorities.
 - Ask about design and implementation sections.
   - design/method is enough to implement. Things like strategy.
     - Mention changes and justify using papers.
   - impl is specific details. Address hDQN.