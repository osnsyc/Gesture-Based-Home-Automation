#!/usr/bin/env python3
from controller import HandController
import requests

def bravia_tv(code):
    # https://pro-bravia.sony.net/zhs/develop/integrate/ircc-ip/overview/index.html
    url = 'http://192.168.1.117/sony/ircc'
    headers = {
        'Accept': '*/*',
        'Content-Type': 'text/xml; charset=UTF-8',
        'SOAPACTION': '"urn:schemas-sony-com:service:IRCC:1#X_SendIRCC"',
        'X-Auth-PSK': 'sony',
        'Connection': 'Keep-Alive',
    }

    """code
    VolumeUp   AAAAAQAAAAEAAAASAw==
    VolumeDown AAAAAQAAAAEAAAATAw==
    Down       AAAAAQAAAAEAAAB1Aw==
    Next       AAAAAgAAAJcAAAA9Aw==
    """
    soap_body = f'''<s:Envelope
        xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
        s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
        <s:Body>
            <u:X_SendIRCC xmlns:u="urn:schemas-sony-com:service:IRCC:1">
                <IRCCCode>{code}</IRCCCode>
            </u:X_SendIRCC>
        </s:Body>
    </s:Envelope>'''
    response = requests.post(url, headers=headers, data=soap_body)

def get_request():
    try:
        response = requests.get('http://HTTP_IN_RED_NODE', timeout=1) # http node in node-red
    except requests.exceptions.RequestException as e:
        print('GET request failed.')
        print('Error:', e)
        return {"error": str(e)}

# Callbacks
def gesture_in_region(event):
    print(f"[Example 1] Your gesture is within a given 3D region!")

def turn_left_turn_right(event):
    rotation = event.hand.rotation
    if rotation > 0.2 and rotation < 0.6:
        print(f"[Example 2] Turn left!")
        # bravia_tv("AAAAAQAAAAEAAAATAw==") # lowing volume down
    elif rotation > -0.6 and rotation < -0.2:
        print(f"[Example 2] Turn right!")
        # bravia_tv("AAAAAQAAAAEAAAASAw==") # rising volume up
    else:
        return

StartToEend = [False, False]
def next_song(event):
    global StartToEend

    if event.hand.rotation > -1.5 and event.hand.rotation < -0.2:
        if StartToEend[0]:
            StartToEend = [False, False]
        else:
            StartToEend[0] = True
    elif event.hand.rotation < 1.5 and event.hand.rotation > 0.2:
        if StartToEend[0]:
            StartToEend[1] = True
        else:
            StartToEend = [False, False]
    else:
        StartToEend = [False, False]

    if StartToEend[0] and StartToEend[1]:
        StartToEend = [False, False]
        print(f"[Example 3] NEXT SONG !")
        # bravia_tv("AAAAAQAAAAEAAAB1Aw==")
    
def making_down(event):
    print(f"[Example 4] You are drawing 'Down'!")

def two_fist(event):
    print(f"[Example 5] Two 'FIST'!")

def hybrid_gestures_with_drawing(event):
    print(f"[Example 6] Your left hand is making an 'OPEN' gesture, while your right hand is making a 'ONE' gesture and moving to the right.")
    # event.print_line()
def up_up(event):
    print(f"[Example 7] Your hands are making a 'TWO' gesture and moving upward.")

blink_data = [0, 0] #[count, frame_nb]
def blink(event):
    global blink_data
    if event.frame_nb - blink_data[1] > 40:
        blink_data = [0, event.frame_nb]
    else:
        blink_data = [blink_data[0]+1, event.frame_nb]

    if blink_data[0] >= 1:
        blink_data[0] = 0
        print(f"[Example 8] Blink!")

TEST_REGION = [0,500,-500,500,0,100] # [x0,x1,y0,y1,z0,z1]
config = {

    'tracker': {'args': {'device_id': 20, 'vector_x': [-1,0,0], 'vector_y': [0,-1,0], 'vector_o': [0,0,0],}},

    'renderer' : {'enable': True, 'args':{'draw_hand': True,'draw_body': True, 'output': None}},

    'debug' : {'enable': False},
    
    'pose_actions' : [
        # Example making gesture in a given region.
        {
            'name': 'gesture_in_region', 
            'hand':'right', 
            'pose':['THREE'], 
            'line': ['all'], 
            'region': TEST_REGION, 
            'callback': 'gesture_in_region',
            "trigger": "enter", 
        },
        # Example slanted gesture.
        {
            'name': 'turn_left_turn_right', 
            'hand':'right', 
            'pose':['TWO'], 
            'line': ['all'], 
            'region': 'all', 
            'callback': 'turn_left_turn_right', 
            "trigger":"periodic",
            "next_trigger_delay": 0.1,
        },
        # Example waving hand from right to left.
        {
            'name': 'next_song', 
            'hand':'right', 
            'pose':['OPEN'], 
            'line': ['all'], 
            'region': 'all', 
            'callback': 'next_song', 
            "trigger":"enter_leave", 
            "max_missing_frames": 15
        },
        # Example making gesture 'ONE' with the right hand and drawing 'Down'.
        {
            'name': 'making_down', 
            'hand':'right', 
            'pose':['ONE'], 
            'line': ['Down'], 
            'region': 'all', 
            'callback': 'making_down',
            "first_trigger_delay": 0.2,
        },
        # Example hybrid gestures
        {
            'name': 'two_fist', 
            'hand':'all', 
            'pose':['FIST', 'FIST'], 
            'line': ['all', 'all'], 
            'region': 'all', 
            'callback': 'two_fist'
        },
        # Example hybrid gestures with drawing
        {
            'name': 'hybrid_gestures_with_drawing', 
            'hand':'all', 
            'pose':['OPEN', 'ONE'], 
            'line': ['all', 'Right'], 
            'region': 'all', 
            'callback': 'hybrid_gestures_with_drawing',
            "trigger": "periodical",
            "next_trigger_delay":0.2,
            "max_missing_frames": 5
        },
        # Example hybrid gestures with hybrid drawing
        {
            'name': 'up_up', 
            'hand':'all', 
            'pose':['TWO', 'TWO'], 
            'line': ['Up', 'Up'], 
            'region': 'all', 
            'callback': 'up_up',
            "trigger": "continuous",
        },
        # Example blink gestures
        {
            'name': 'blink', 
            'hand':'left', 
            'pose':['OPEN'], 
            'line': ['all'], 
            'region': 'all', 
            'callback': 'blink',
            "trigger": "enter",
            "first_trigger_delay": 0.1, 
            "max_missing_frames": 5,
        },
    ]
}

HandController(config).loop()
