# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (symb) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File res: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method


@lru_cache(maxsize=100)  # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable res.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable res, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
        (len(string) == 1 or string[1:].isdecimal())


@lru_cache(maxsize=100)  # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'


@lru_cache(maxsize=100)  # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'


@lru_cache(maxsize=100)  # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'
    # For Chapter 3:
    # return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}


@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable res, or operator at the root of
            the formula tree.
        now (`~typing.Optional`\\[`Formula`]): the now operand of the root,
            if the root is a unary or binary operator.
        next (`~typing.Optional`\\[`Formula`]): the next operand of the
            root, if the root is a binary operator.
    """
    root: str
    now: Optional[Formula]
    next: Optional[Formula]

    def __init__(self, root: str, now: Optional[Formula] = None,
                 next: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            now: the now operand for the root, if the root is a unary or
                binary operator.
            next: the next operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert now is None and next is None
            self.root = root
        elif is_unary(root):
            assert now is not None and next is None
            self.root, self.now = root, now
        else:
            assert is_binary(root)
            assert now is not None and next is not None
            self.root, self.now, self.next = root, now, next

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 1.1
        if is_unary(self.root):
            return self.root + repr(self.now)
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        
        return '(' + repr(self.now) + self.root + repr(self.next) + ')'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 1.2
        if is_constant(self.root):
            return set()
        if is_variable(self.root):
            return {self.root}
        
        if is_unary(self.root):
            return self.now.variables()
        return self.now.variables() | self.next.variables()

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        # Task 1.3
        if is_variable(self.root):
            return set()
        if is_constant(self.root):
            return {self.root}
        if is_unary(self.root):
            return {self.root} | self.now.operators()
        return {self.root} | self.now.operators() | self.next.operators()

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable res (e.g.,
            ``'x12'``) or a unary operator followed by a variable res, then the
            parsed prefix will include that entire variable res (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        # Task 1.4
        if string == '':
            return None, 'End of input'
        sym = string[0]
        if is_constant(sym):
            return Formula(sym), string[1:]
        if sym == '(':
            now, ost = Formula._parse_prefix(string[1:])
            if now is None:
                return None, ost
            if ost == '':
                return None, 'Missing operator'
            chr = None
            for i in ['<->', '-&', '->', '-|', '&', '|', '+']:
                if ost.startswith(i):
                    chr = i
                    ost = ost[len(i):]
                    break
            if chr is None:
                return None, 'Invalid operator'
            if not is_binary(chr):
                return None, 'Invalid operator'
            next, ost = Formula._parse_prefix(ost)
            if next is None:
                return None, ost
            if ost == '' or ost[0] != ')':
                return None, 'Not closed'
            return Formula(chr, now, next), ost[1:]
    
        if sym == '~':
            leaf, ost = Formula._parse_prefix(string[1:])
            if leaf is None:
                return None, ost
            return Formula('~', leaf), ost

        if 'p' <= sym <= 'z':
            num = 1
            while num < len(string) and string[num].isdecimal():
                num += 1
            res = string[:num]
            if not is_variable(res):
                return None, 'Invalid'
            return Formula(res), string[num:]
        return None, 'Invalid pref'

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        # Task 1.5
        formula, ost = Formula._parse_prefix(string)
        fl = formula is not None and ost == ''
        return fl

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        # Task 1.6
        formula, ost = Formula._parse_prefix(string)
        fl = formula is not None and ost == ''
        assert fl
        return formula

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        if is_unary(self.root):
            return self.root + self.now.polish()
        return self.root + self.now.polish() + self.next.polish()

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8
        res = []
        i = 0
        while i < len(string):
            symb = string[i]
            if symb == 'T' or symb == 'F':
                res.append(symb)
                i += 1
                continue
            
            if symb == '~':
                res.append('~')
                i += 1
                continue
            if symb in {'&', '|', '+'}:
                res.append(symb)
                i += 1
                continue

            if string.startswith('<->', i):
                res.append('<->')
                i += 3
                continue
            if string.startswith('->', i):
                res.append('->')
                i += 2
                continue
            if string.startswith('-&', i):
                res.append('-&')
                i += 2
                continue
            if string.startswith('-|', i):
                res.append('-|')
                i += 2
                continue

            if 'p' <= symb <= 'z':
                next = i + 1
                while next < len(string) and string[next].isdecimal():
                    next += 1
                leaf = string[i:next]
                if not is_variable(leaf):
                    raise ValueError('Invalid variable')
                res.append(leaf)
                i = next
                continue

            raise ValueError('Invalid symbol')
        res_string = []
        for el in reversed(res):
            if is_constant(el) or is_variable(el):
                res_string.append(Formula(el))
            elif is_binary(el):
                if len(res_string) < 2:
                    raise ValueError('Invalid')
                now = res_string.pop()
                next = res_string.pop()
                res_string.append(Formula(el, now, next))
            elif is_unary(el):
                if len(res_string) < 1:
                    raise ValueError('Invalid')
                leaf = res_string.pop()
                res_string.append(Formula(el, leaf))
            else:
                raise ValueError('Invalid')
        if len(res_string) != 1:
            raise ValueError('Invalid')
        return res_string[0]

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable res `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable res occurrences originating in the current formula are
            substituted (i.e., variable res occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `chr`
        that is a key in `substitution_map` with the formula
        `substitution_map[chr]` applied to its (zero or one or two) operands,
        where the now operand is used for every occurrence of ``'p'`` in the
        formula and the next for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
