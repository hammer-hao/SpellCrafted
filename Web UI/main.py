from browser import ajax, document, bind
from browser.html import *


def card_container(img_path, con_id, card_w=969, card_h=1352, card_display_h=50):
    card_con = DIV(id=con_id, Class='card_container',
                   style={'width': f'calc({card_display_h}vh * {card_w} / {card_h})',
                          'height': f'{card_display_h}vh',
                          'background-image': f"url({img_path})"})
    return card_con


card_pos = DIV(Class='card_pos')

card_gen = card_container('img/card_template.png', 'card_gen')
card_gen <= DIV(id='card_name') + DIV(id='card_cost') + DIV(id='card_des')

card_back = card_container('img/card_back.png', 'card_back')
card_pos <= card_back
document <= card_pos

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


def show_generated_card(resp):
    print(resp.json)
    resp = resp.json

    card_back.style.display = 'none'
    card_pos <= card_gen
    card_gen.style.display = 'initial'
    document['card_name'].text = resp['name']
    document['card_cost'].text = resp['cost']
    document['card_des'].text = resp['description']

    generating.text = 'Reset'
    generating.classList.add('reset')

    @bind(generating, 'click')
    def reset_card(ev):
        card_gen.style.display = 'none'
        card_back.style.display = 'initial'
        ev.target.remove()
        ev.target.unbind('click', reset_card)

