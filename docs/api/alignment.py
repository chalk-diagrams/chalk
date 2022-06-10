from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))

# ### Diagram.show_origin

# + tags=["hide_inp"]
help(Diagram.show_origin)
# -

#

triangle(1).show_origin()

# ### Diagram.show_bounding_box

# + tags=["hide_inp"]
help(Diagram.show_bounding_box)
# -

#

triangle(1).show_bounding_box()

# ### Diagram.align_t

# + tags=["hide_inp"]
help(Diagram.align_t)
# -

#

triangle(1).align_t().show_bounding_box().show_origin()

# ### Diagram.align_r

# + tags=["hide_inp"]
help(Diagram.align_r)
# -

#

triangle(1).align_r().show_bounding_box().show_origin()

# ### Diagram.center_xy

# + tags=["hide_inp"]
help(Diagram.center_xy)
# -

#

triangle(1).center_xy().show_bounding_box().show_origin()


# ### Diagram.pad

# + tags=["hide_inp"]
help(Diagram.pad)
# -

#

triangle(1).pad(0.5).show_bounding_box().show_origin()


