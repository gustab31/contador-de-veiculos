# Identificador de veiculos "tracker"

import math

class EuclideanDistTracker:
    def __init__(self):
        # Armazena a posicao de centro dos objetos
        self.centros = {}
        # mantem a contagem
        # cada contagem um novo obj é detectado aumentando um
        self.contador = 0


    def update(self, objetos_retan):
        # Caixa de objetos 'boxes'
        objetos_ident = []

        # Ponto cetral de um novo objeto
        for retang in objetos_retan:
            x, y, w, h, index = retang
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Verifica se o objeto já foi detectado
            ja_detectado = False
            for id, pt in self.centros.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.centros[id] = (cx, cy)
                    # print(self.centros)
                    objetos_ident.append([x, y, w, h, id, index])
                    ja_detectado = True
                    break

            # Novo objeto é detectado e contado
            if ja_detectado is False:
                self.centros[self.contador] = (cx, cy)
                objetos_ident.append([x, y, w, h, self.contador, index])
                self.contador += 1

        # Limpa a lista de pontos centrais 
        novo_centro = {}
        for obj_box_ident in objetos_ident:
            _, _, _, _, object_id, index = obj_box_ident
            center = self.centros[object_id]
            novo_centro[object_id] = center

        # Atualiza a lista
        self.centros = novo_centro.copy()
        return objetos_ident


# Retorna
def ad(a, b):
    return a+b