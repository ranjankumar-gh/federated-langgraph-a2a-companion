"""The partner stub: a stand-in A2A server for a partner the caller cannot deploy.

It is built only from what the caller holds about the partner: the card it fetched,
the partner's terms (published on the card or agreed on paper), recorded tasks, and a
fault script. Nothing in this package imports the partner's code, and a test holds it
to that.
"""
