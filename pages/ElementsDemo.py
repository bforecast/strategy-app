# First, import the elements you need

from streamlit_elements import elements, mui, html

# Create a frame where Elements widgets will be displayed.
#
# Elements widgets will not render outside of this frame.
# Native Streamlit widgets will not render inside this frame.
#
# elements() takes a key as parameter.
# This key can't be reused by another frame or Streamlit widget.

with elements("new_element"):

    # Let's create a Typography element with "Hello world" as children.
    # The first step is to check Typography's documentation on MUI:
    # https://mui.com/components/typography/
    #
    # Here is how you would write it in React JSX:
    #
    # <Typography>
    #   Hello world
    # </Typography>

    mui.Typography("Hello world")

with elements("multiple_children"):

    # You have access to Material UI icons using: mui.icon.IconNameHere
    #
    # Multiple children can be added in a single element.
    #
    # <Button>
    #   <EmojiPeople />
    #   <DoubleArrow />
    #   Hello world
    # </Button>

    mui.Button(
        mui.icon.EmojiPeople,
        mui.icon.DoubleArrow,
        "Button with multiple children"
    )

    # You can also add children to an element using a 'with' statement.
    #
    # <Button>
    #   <EmojiPeople />
    #   <DoubleArrow />
    #   <Typography>
    #     Hello world
    #   </Typography>
    # </Button>

    with mui.Button:
        mui.icon.EmojiPeople()
        mui.icon.DoubleArrow()
        mui.Typography("Button with multiple children")    

with elements("nested_children"):

    # You can nest children using multiple 'with' statements.
    #
    # <Paper>
    #   <Typography>
    #     <p>Hello world</p>
    #     <p>Goodbye world</p>
    #   </Typography>
    # </Paper>

    with mui.Paper:
        with mui.Typography:
            html.p("Hello world")
            html.p("Goodbye world")

with elements("properties"):

    # You can add properties to elements with named parameters.
    #
    # To find all available parameters for a given element, you can
    # refer to its related documentation on mui.com for MUI widgets,
    # on https://microsoft.github.io/monaco-editor/ for Monaco editor,
    # and so on.
    #
    # <Paper elevation={3} variant="outlined" square>
    #   <TextField label="My text input" defaultValue="Type here" variant="outlined" />
    # </Paper>

    with mui.Paper(elevation=3, variant="outlined", square=True):
        mui.TextField(
            label="My text input",
            defaultValue="Type here",
            variant="outlined",
        )

    # If you must pass a parameter which is also a Python keyword, you can append an
    # underscore to avoid a syntax error.
    #
    # <Collapse in />

    mui.Collapse(in_=True)

    # mui.collapse(in=True)
    # > Syntax error: 'in' is a Python keyword:            

with elements("style_mui_sx"):

    # For Material UI elements, use the 'sx' property.
    #
    # <Box
    #   sx={{
    #     bgcolor: 'background.paper',
    #     boxShadow: 1,
    #     borderRadius: 2,
    #     p: 2,
    #     minWidth: 300,
    #   }}
    # >
    #   Some text in a styled box
    # </Box>

    mui.Box(
        "Some text in a styled box",
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "minWidth": 300,
        }
    )    

with elements("style_elements_css"):

    # For any other element, use the 'css' property.
    #
    # <div
    #   css={{
    #     backgroundColor: 'hotpink',
    #     '&:hover': {
    #         color: 'lightgreen'
    #     }
    #   }}
    # >
    #   This has a hotpink background
    # </div>

    html.div(
        "This has a hotpink background",
        css={
            "backgroundColor": "hotpink",
            "&:hover": {
                "color": "lightgreen"
            }
        }
    )    

    import streamlit as st

with elements("callbacks_retrieve_data"):

    # Some element allows executing a callback on specific event.
    #
    # const [name, setName] = React.useState("")
    # const handleChange = (event) => {
    #   // You can see here that a text field value
    #   // is stored in event.target.value
    #   setName(event.target.value)
    # }
    #
    # <TextField
    #   label="Input some text here"
    #   onChange={handleChange}
    # />

    # Initialize a new item in session state called "my_text"
    if "my_text" not in st.session_state:
        st.session_state.my_text = ""

    # When text field changes, this function will be called.
    # To know which parameters are passed to the callback,
    # you can refer to the element's documentation.
    def handle_change(event):
        st.session_state.my_text = event.target.value

    # Here we display what we have typed in our text field
    mui.Typography(st.session_state.my_text)

    # And here we give our 'handle_change' callback to the 'onChange'
    # property of the text field.
    mui.TextField(label="Input some text here", onChange=handle_change)

with elements("callbacks_sync"):

    # If you just want to store callback parameters into Streamlit's session state
    # like above, you can also use the special function sync().
    #
    # When an onChange event occurs, the callback is called with an event data object
    # as argument. In the example below, we are synchronizing that event data object with
    # the session state item 'my_event'.
    #
    # If an event passes more than one parameter, you can synchronize as many session state item
    # as needed like so:
    # >>> sync("my_first_param", "my_second_param")
    #
    # If you want to ignore the first parameter of an event but keep synchronizing the second,
    # pass None to sync:
    # >>> sync(None, "second_parameter_to_keep")

    from streamlit_elements import sync

    if "my_event" not in st.session_state:
        st.session_state.my_event = None

    if st.session_state.my_event is not None:
        text = st.session_state.my_event.target.value
    else:
        text = ""

    mui.Typography(text)
    mui.TextField(label="Input some text here", onChange=sync("my_event"))    


with elements("callbacks_lazy"):

    # With the two first examples, each time you input a letter into the text field,
    # the callback is invoked but the whole app is reloaded as well.
    #
    # To avoid reloading the whole app on every input, you can wrap your callback with
    # lazy(). This will defer the callback invocation until another non-lazy callback
    # is invoked. This can be useful to implement forms.

    from streamlit_elements import lazy

    if "first_name" not in st.session_state:
        st.session_state.first_name = None
        st.session_state.last_name = None

    if st.session_state.first_name is not None:
        first_name = st.session_state.first_name.target.value
    else:
        first_name = "John"

    if st.session_state.last_name is not None:
        last_name = st.session_state.last_name.target.value
    else:
        last_name = "Doe"

    def set_last_name(event):
        st.session_state.last_name = event

    # Display first name and last name
    mui.Typography("Your first name: ", first_name)
    mui.Typography("Your last name: ", last_name)

    # Lazily synchronize onChange with first_name and last_name state.
    # Inputting some text won't synchronize the value yet.
    mui.TextField(label="First name", onChange=lazy(sync("first_name")))

    # You can also pass regular python functions to lazy().
    mui.TextField(label="Last name", onChange=lazy(set_last_name))

    # Here we give a non-lazy callback to onClick using sync().
    # We are not interested in getting onClick event data object,
    # so we call sync() with no argument.
    #
    # You can use either sync() or a regular python function.
    # As long as the callback is not wrapped with lazy(), its invocation will
    # also trigger every other defered callbacks.
    mui.Button("Update first namd and last name", onClick=sync())    

with elements("callbacks_hotkey"):

    # Invoke a callback when a specific hotkey sequence is pressed.
    #
    # For more information regarding sequences syntax and supported keys,
    # go to Mousetrap's project page linked below.
    #
    # /!\ Hotkeys work if you don't have focus on Streamlit Elements's frame /!\
    # /!\ As with other callbacks, this reruns the whole app /!\

    from streamlit_elements import event

    def hotkey_pressed():
        print("Hotkey pressed")

    event.Hotkey("g", hotkey_pressed)

    # If you want your hotkey to work even in text fields, set bind_inputs to True.
    event.Hotkey("h", hotkey_pressed, bindInputs=True)
    mui.TextField(label="Try pressing 'h' while typing some text here.")

    # If you want to override default hotkeys (ie. ctrl+f to search in page),
    # set overrideDefault to True.
    event.Hotkey("ctrl+f", hotkey_pressed, overrideDefault=True)    

with elements("dashboard"):

    # You can create a draggable and resizable dashboard using
    # any element available in Streamlit Elements.

    from streamlit_elements import dashboard

    # First, build a default layout for every element you want to include in your dashboard

    layout = [
        # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
        dashboard.Item("first_item", 0, 0, 2, 2),
        dashboard.Item("second_item", 2, 0, 2, 2, isDraggable=False, moved=False),
        dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
    ]

    # Next, create a dashboard layout using the 'with' syntax. It takes the layout
    # as first parameter, plus additional properties you can find in the GitHub links below.

    with dashboard.Grid(layout):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")

    # If you want to retrieve updated layout values as the user move or resize dashboard items,
    # you can pass a callback to the onLayoutChange event parameter.

    def handle_layout_change(updated_layout):
        # You can save the layout in a file, or do anything you want with it.
        # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        print(updated_layout)

    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")

with elements("monaco_editors"):

    # Streamlit Elements embeds Monaco code and diff editor that powers Visual Studio Code.
    # You can configure editor's behavior and features with the 'options' parameter.
    #
    # Streamlit Elements uses an unofficial React implementation (GitHub links below for
    # documentation).

    from streamlit_elements import editor

    if "content" not in st.session_state:
        st.session_state.content = "Default value"

    mui.Typography("Content: ", st.session_state.content)

    def update_content(value):
        st.session_state.content = value

    editor.Monaco(
        height=300,
        defaultValue=st.session_state.content,
        onChange=lazy(update_content)
    )

    mui.Button("Update content", onClick=sync())

    editor.MonacoDiff(
        original="Happy Streamlit-ing!",
        modified="Happy Streamlit-in' with Elements!",
        height=300,
    ) 


with elements("nivo_charts"):

    # Streamlit Elements includes 45 dataviz components powered by Nivo.

    from streamlit_elements import nivo

    DATA = [
        { "taste": "fruity", "chardonay": 93, "carmenere": 61, "syrah": 114 },
        { "taste": "bitter", "chardonay": 91, "carmenere": 37, "syrah": 72 },
        { "taste": "heavy", "chardonay": 56, "carmenere": 95, "syrah": 99 },
        { "taste": "strong", "chardonay": 64, "carmenere": 90, "syrah": 30 },
        { "taste": "sunny", "chardonay": 119, "carmenere": 94, "syrah": 103 },
    ]

    with mui.Box(sx={"height": 500}):
        nivo.Radar(
            data=DATA,
            keys=[ "chardonay", "carmenere", "syrah" ],
            indexBy="taste",
            valueFormat=">-.2f",
            margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
            borderColor={ "from": "color" },
            gridLabelOffset=36,
            dotSize=10,
            dotColor={ "theme": "background" },
            dotBorderWidth=2,
            motionConfig="wobbly",
            legends=[
                {
                    "anchor": "top-left",
                    "direction": "column",
                    "translateX": -50,
                    "translateY": -40,
                    "itemWidth": 80,
                    "itemHeight": 20,
                    "itemTextColor": "#999",
                    "symbolSize": 12,
                    "symbolShape": "circle",
                    "effects": [
                        {
                            "on": "hover",
                            "style": {
                                "itemTextColor": "#000"
                            }
                        }
                    ]
                }
            ],
            theme={
                "background": "#FFFFFF",
                "textColor": "#31333F",
                "tooltip": {
                    "container": {
                        "background": "#FFFFFF",
                        "color": "#31333F",
                    }
                }
            }
        )           

with elements("data_grid"):

    # Streamlit Elements includes 45 dataviz components powered by Nivo.

    from streamlit_elements import datagrid

    DATA = [
        { "taste": "fruity", "chardonay": 93, "carmenere": 61, "syrah": 114 },
        { "taste": "bitter", "chardonay": 91, "carmenere": 37, "syrah": 72 },
        { "taste": "heavy", "chardonay": 56, "carmenere": 95, "syrah": 99 },
        { "taste": "strong", "chardonay": 64, "carmenere": 90, "syrah": 30 },
        { "taste": "sunny", "chardonay": 119, "carmenere": 94, "syrah": 103 },
    ]

    with mui.Box(sx={"height": 500}):
        datagrid(
            data=DATA,
            keys=[ "chardonay", "carmenere", "syrah" ],
            indexBy="taste",
            valueFormat=">-.2f",
            margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
            borderColor={ "from": "color" },
            gridLabelOffset=36,
            dotSize=10,
            dotColor={ "theme": "background" },
            dotBorderWidth=2,
            motionConfig="wobbly",
            legends=[
                {
                    "anchor": "top-left",
                    "direction": "column",
                    "translateX": -50,
                    "translateY": -40,
                    "itemWidth": 80,
                    "itemHeight": 20,
                    "itemTextColor": "#999",
                    "symbolSize": 12,
                    "symbolShape": "circle",
                    "effects": [
                        {
                            "on": "hover",
                            "style": {
                                "itemTextColor": "#000"
                            }
                        }
                    ]
                }
            ],
            theme={
                "background": "#FFFFFF",
                "textColor": "#31333F",
                "tooltip": {
                    "container": {
                        "background": "#FFFFFF",
                        "color": "#31333F",
                    }
                }
            }
        )                   