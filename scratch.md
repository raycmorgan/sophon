```
[foo]

    [\0, foo]
    /      \
[bar]      [foo]
```
```
[foo]

    [\0, zoo]
    /      \
[foo]      [zoo]
```

Special case:
    If pivot = 0, use larger of new_key & pivot_key as parent's pointer.

```

    [\0, foo]
    /       \
[bar, door] [foo, moo, zoo]

            Split:  [foo] [moo, zoo]
                Pivot = 3 / 2 => 1
                    (Move everything >= pivot to right)
            Insert: [foo, koo] [moo, zoo]


    [\0,       foo,      moo]
    /           |           \
[bar, door] [foo, koo] [moo, zoo]

```

1. Split the child at the mid point.
2. Insert the key
3. Set the right (new node's) lower_fence to the 





```
[foo]

     [(,foo]].∞
    /          \
[bar]         [foo]
```

```
[foo]

     [(,foo]].∞
    /          \
[bar]         [foo]
```
