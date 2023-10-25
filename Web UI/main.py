from browser import ajax, document, bind
from browser.html import *
import re
import random

card_w = 969
card_h = 1352
card_display_h = 50


def card_container(img_path, con_id):
    card_con = DIV(id=con_id, Class='card_container',
                   style={'background-image': f"url({img_path})"})
    return card_con


card_pos = DIV(Class='card_pos', style={'width': f'calc({card_display_h}vh * {card_w} / {card_h})',
                                        'height': f'{card_display_h}vh'})

card_gen = card_container('img/card_template.png', 'card_gen')
card_title = DIV(id='card_title')
card_title <= DIV(id='card_name') + DIV(id='card_mana')
card_gen <= card_title + DIV(id='card_cost') + DIV(id='card_des')
card_pos <= card_gen

back_img = ['card_back.png', 'card_back_sun.jpg', 'card_back_mtg.jpg']
back_img_choice = random.randint(0, 2)
card_back = card_container(f'img/{back_img[back_img_choice]}', 'card_back')
card_pos <= card_back
document <= card_pos

# Initialize special style
document['card_back'].classList.add('pointer')
document['card_gen'].classList.add('flip_card')

prompt_con = DIV(Class='prompt_container')
prompt_con <= INPUT(id='prompt', Class='prompt_input')
# prompt_con <= DIV('Generate', Class='generate')

document <= prompt_con

generating = DIV(Class='generating_icon')


@bind(card_back, 'click')
def generate(ev):
    ajax.get('http://35.170.169.132/',
             data={'prompt': document['prompt'].value},
             mode='json',
             cache=True,
             oncomplete=show_generated_card)
    document <= generating
    generating.text = 'Generating...'
    generating.classList.remove('reset')
    document['card_des'].innerHTML = ''
    document['card_mana'].innerHTML = ''


def show_generated_card(resp):
    resp = resp.json
    if resp is None:
        generating.text = '寄'
    else:
        card_pos.classList.add('flip_card')
        print(resp)

        document['card_name'].text = resp['name']
        mana = resp['mana'].replace('{', '').replace('}', '').lower()
        for m in mana:
            document['card_mana'] <= IMG(src=f'mana_icon/mana-{m}.png', Class='icon')
        if mana[-1].isalpha():
            document['card_gen'].style.backgroundImage = f"url(card/{mana[-1]}.jpg)"
        else:
            document['card_gen'].style.backgroundImage = f"url(img/card_template.png)"
        document['card_cost'].text = resp['cost']
        # print(len(resp['cost']), document['card_cost'].width / 16)
        document['card_cost'].style.fontSize = f"{document['card_cost'].height / document['card_cost'].scrollHeight * 100}%"

        des = re.sub('\{.\}', '#', resp['description']).split('#')
        des_icon = re.findall('\{.\}', resp['description'])
        des_con = document['card_des']
        print(des)
        print(des_icon)
        print()
        # des_con <= SPAN("Destroy target creature with power 4 or greater. Its controller may return all nonland cards from your graveyard to the battlefield. 4: Look at the top X cards of your library equal to the number of creature cards revealed this turn. You may put that many + 1 /+ 1 counters on it instead. If that card has no mana value 4 or greater than put into your graveyard from the top of your library instead Horde ' s graveyard: Choose a card name. You may look at the top card of your library. Draw 2 cards, then put it into your hand. If it shares a spell from your hand, put the rest into your hand and the rest on the bottom of your library.) Whenever you draw up to one target creature deals damage to a player, you draw two cards.")
        for s in range(len(des)):
            if s >= 1:
                des_con <= IMG(src=f'mana_icon/mana-{des_icon[s - 1][1].lower()}.png', Class='icon')
            des_con <= SPAN(des[s])

        # des_con.style.fontSize = f"{des_con.height /
        # (-0.002285 * des_con.scrollHeight ** 2 + 1.4979 * des_con.scrollHeight - 26.6) * 16}px "
        des_con.style.fontSize = f"{des_con.height / des_con.scrollHeight * 100}%"
        generating.text = 'Reset'
        generating.classList.add('reset')

        @bind(generating, 'click')
        def reset_card(ev):
            ev.target.remove()
            ev.target.unbind('click', reset_card)
            card_pos.classList.remove('flip_card')


@bind(document['prompt'], 'keydown')
def capitalize_input(ev):
    ev.target.value = ev.target.value.capitalize()
