#!/usr/bin/env python3

class TreeVisitor:
    def leaf(self, v): pass
    def node(self, l, r): pass

class Tree:
    def internal(self, visitor: TreeVisitor): pass
    def external(self, visitor: TreeVisitor): pass

class Leaf(Tree):
    def __init__(self, v):
        self.v = v
    def internal(self, visitor: TreeVisitor):
        return visitor.leaf(self.v)
    def external(self, visitor: TreeVisitor):
        return visitor.leaf(self.v)

class Node(Tree):
    def __init__(self, l, r):
        self.l = l
        self.r = r
    def internal(self, visitor: TreeVisitor):
        return visitor.node(self.l.internal(visitor), self.r.internal(visitor))
    def external(self, visitor: TreeVisitor):
        return visitor.node(self.l, self.r)

class LowerCasePrinter(TreeVisitor):
    def leaf(self, v):
        return f'leaf({v})'
    def node(self, l, r):
        return f'node({l}, {r})'

class UpperCasePrinter(TreeVisitor):
    def leaf(self, v):
        return f'LEAF({v})'
    def node(self, l, r):
        return f'NODE({l}, {r})'

# Allowing the script to choose between different interpretations.
class UpperCase(Tree):
    def __init__(self, tree):
        self.tree = tree
    def internal(self, visitor):
        return visitor.upperCase(self.tree.internal(visitor))
    def external(self, visitor):
        return visitor.upperCase(self.tree)

class LowerCase(Tree):
    def __init__(self, tree):
        self.tree = tree
    def internal(self, visitor):
        return visitor.lowerCase(self.tree.internal(visitor))
    def external(self, visitor):
        return visitor.lowerCase(self.tree)

# External visitor because we need to push down information, the mode, from
# root to leaves.
#
# I think this visitor is monadic, in a way, because the interpretation depends
# on user input. Or maybe `Selective`, as the name hints. Not quite monadic
# because the possible computations from which we select based on user input is
# fixed.
class Dispatching(TreeVisitor):
    def __init__(self):
        self.modes = {
            'lower': LowerCasePrinter(),
            'upper': UpperCasePrinter(),
        }
    def leaf(self, v):
        return lambda mode: self.modes[mode].leaf(v)
    def node(self, l, r):
        return lambda mode: self.modes[mode].node(
            l.external(self)(mode),
            r.external(self)(mode),
        )
    def upperCase(self, t):
        return lambda _: t.external(self)('upper')
    def lowerCase(self, t):
        return lambda _: t.external(self)('lower')

# This visitor is neither external, nor internal. It just delegates to the
# inner `Dispatching` visitor and supplies the default/start_mode at the end.
# However, it must be used as an external visitor because `Dispatching` is
# external.
class SwitchingCase(TreeVisitor):
    def __init__(self):
        self.dispatching = Dispatching()
        self.start_mode = 'lower'
    def leaf(self, v):
        return self.dispatching.leaf(v)(self.start_mode)
    def node(self, l, r):
        return self.dispatching.node(l, r)(self.start_mode)
    def upperCase(self, t):
        return self.dispatching.upperCase(t)(self.start_mode)
    def lowerCase(self, t):
        return self.dispatching.lowerCase(t)(self.start_mode)

tree = Node(
    Node(
        Leaf(1),
        Node(Leaf(2), Leaf(3)),
    ),
    Node(Leaf(4), Leaf(5)),
)

print(tree.internal(LowerCasePrinter()))
print(tree.internal(UpperCasePrinter()))

tree = Node(
    UpperCase(Node(
        LowerCase(Leaf(1)),
        Node(Leaf(2), Leaf(3)),
    )),
    Node(Leaf(4), Leaf(5)),
)

print(tree.external(Dispatching())('upper'))
print(tree.external(Dispatching())('lower'))
print(tree.external(SwitchingCase()))
