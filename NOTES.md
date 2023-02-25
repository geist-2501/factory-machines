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
   - B A E F D C <- depots