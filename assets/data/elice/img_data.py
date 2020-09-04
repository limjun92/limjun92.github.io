import numpy as np

img_data = np.array([[2.82292702e-01, 5.53599751e-01, 3.13360365e-01, 6.85218093e-01,
        1.19928514e-01, 9.77551662e-01, 8.30417148e-01, 3.76009489e-01,
        2.02970095e-01, 1.78365232e-01, 3.27735206e-01, 2.67258168e-01,
        5.13403358e-02, 1.20852645e-01, 2.45525317e-01, 9.52467214e-01,
        7.82066323e-03, 6.83325136e-01, 7.28755124e-01, 7.37231835e-01,
        9.60197038e-01, 2.00383338e-01, 3.86716309e-02, 5.17772291e-01,
        4.82777210e-02, 9.89531416e-01, 1.46952405e-01, 4.07697116e-01,
        9.92821383e-01, 1.73763925e-01, 3.23919312e-02, 8.74562231e-01,
        2.00297105e-01, 2.34659858e-01, 1.23572878e-01, 1.09020709e-01,
        7.12795671e-01, 1.06566111e-01, 7.38437750e-01, 8.17081831e-01,
        2.51034331e-01, 2.14599703e-01, 3.33028528e-01, 4.46268351e-02],
       [4.94579248e-01, 1.92357252e-02, 2.42078217e-01, 4.70780482e-01,
        2.84317951e-02, 3.54364458e-01, 4.20156699e-01, 7.37035442e-01,
        4.94638871e-01, 9.80492661e-01, 4.73130644e-01, 1.17224428e-01,
        3.25756354e-01, 6.55773949e-01, 6.55530393e-01, 6.17410595e-01,
        8.64407794e-01, 2.79972041e-01, 4.35883229e-01, 7.90383169e-01,
        3.93214703e-02, 7.97563708e-01, 6.86027938e-01, 6.99704358e-02,
        1.85614977e-01, 2.06149186e-01, 8.62229901e-02, 2.92922594e-01,
        3.99652545e-01, 2.51742811e-01, 8.01991783e-01, 6.03476047e-03,
        9.41721561e-01, 1.40959291e-01, 7.13672519e-03, 5.82649523e-01,
        4.89155384e-01, 8.62510051e-01, 1.49298724e-01, 5.14150463e-01,
        4.22197383e-01, 7.25673639e-01, 9.22671640e-01, 7.05184672e-01],
       [5.29735639e-01, 9.82397171e-01, 2.49884364e-01, 2.20250555e-01,
        4.98503887e-01, 7.92641585e-01, 8.73776712e-01, 5.28869046e-02,
        5.91330258e-01, 1.72083707e-01, 5.47788001e-01, 3.07527976e-02,
        4.14455031e-01, 9.05228162e-01, 9.76061253e-01, 6.39310572e-01,
        9.26367605e-01, 3.18593299e-01, 5.00146844e-01, 7.22147039e-02,
        1.39261015e-01, 3.53669466e-02, 8.48459065e-01, 1.57337207e-01,
        5.75826564e-01, 7.16081481e-01, 2.60830556e-01, 7.57921374e-01,
        1.29981738e-01, 5.02486761e-01, 5.02781461e-01, 7.08395788e-02,
        8.37747666e-01, 8.26042246e-01, 9.25356961e-01, 8.10515955e-01,
        7.95153147e-01, 8.72251016e-01, 4.82239511e-01, 6.30973514e-01,
        2.08103145e-01, 3.85505209e-01, 1.87462886e-01, 6.48147234e-01],
       [6.88669639e-01, 2.47168694e-01, 7.74264506e-03, 7.83075967e-01,
        2.09675114e-01, 9.60021314e-01, 1.78188026e-01, 4.98018978e-01,
        8.25458220e-01, 9.31582589e-01, 8.25973096e-01, 6.75428697e-01,
        5.63538650e-01, 5.89619641e-01, 2.01778684e-01, 7.73053281e-01,
        2.45176098e-01, 9.59428144e-01, 4.20316365e-01, 6.91808223e-02,
        5.93189081e-01, 6.54163880e-01, 5.18094728e-01, 9.85394692e-01,
        9.07346667e-01, 7.86049450e-01, 8.36158055e-01, 5.10890369e-01,
        7.22949535e-01, 9.51836848e-01, 1.62842513e-02, 7.14537802e-01,
        9.26341384e-01, 9.11498694e-01, 9.79292850e-01, 8.99924480e-01,
        3.76963717e-01, 6.41508088e-01, 6.41940323e-01, 6.43151499e-01,
        9.44629369e-01, 5.46951795e-03, 3.30390641e-01, 6.80777458e-01],
       [3.42150019e-01, 6.87871430e-01, 1.88423539e-01, 2.42284039e-01,
        2.42772464e-01, 3.73582286e-01, 6.73803256e-01, 1.85391862e-01,
        4.77956520e-01, 3.27007329e-01, 1.59633090e-01, 4.36740598e-01,
        3.16867821e-01, 6.11562449e-01, 1.92752359e-01, 3.26240675e-02,
        2.86947570e-01, 3.79434091e-01, 4.55614988e-01, 2.53157806e-02,
        2.52315267e-01, 4.97352505e-01, 4.34677759e-01, 9.13338825e-01,
        1.31337117e-01, 9.83565034e-01, 9.20114209e-01, 4.37603531e-02,
        9.52034679e-01, 6.14393190e-01, 4.64136715e-01, 7.93342484e-01,
        6.23625658e-01, 7.48349020e-01, 8.17243290e-01, 7.70664470e-01,
        6.77808266e-01, 1.44163886e-01, 6.39123214e-01, 7.49827587e-01,
        1.52531860e-02, 2.40769192e-01, 1.75202651e-02, 8.68347407e-01],
       [1.26559632e-01, 2.99558933e-01, 9.69081001e-01, 6.84096213e-02,
        4.89089240e-01, 5.95044679e-01, 2.25327555e-01, 9.33613781e-01,
        2.05522867e-01, 1.64450885e-02, 7.29168400e-01, 6.11931879e-01,
        8.41135697e-01, 2.10123720e-01, 8.10688181e-01, 4.45488075e-01,
        9.77864482e-01, 8.95318006e-01, 7.66171752e-02, 9.23967116e-01,
        3.94590068e-01, 6.88192938e-01, 6.20201916e-01, 9.16666528e-01,
        2.39440600e-02, 6.97400004e-01, 5.04599125e-01, 1.16126692e-01,
        5.54681129e-01, 9.42919021e-01, 6.84409400e-01, 1.14138180e-01,
        4.55197040e-01, 5.03334332e-02, 6.18530952e-01, 9.41498072e-01,
        4.94144374e-01, 9.16260852e-01, 7.69886287e-01, 9.01527073e-01,
        7.82197428e-01, 8.05590994e-01, 6.21149102e-01, 5.84490443e-01],
       [1.21779048e-01, 8.21005101e-01, 7.47526487e-01, 5.47051814e-01,
        5.62114588e-01, 6.13698562e-01, 3.23728906e-01, 9.22469063e-01,
        4.86487958e-01, 9.18958013e-01, 9.67130033e-01, 7.91053765e-01,
        1.90407103e-01, 6.04228833e-01, 2.75682043e-01, 5.02258959e-01,
        8.94939223e-01, 3.50961197e-01, 7.48701169e-01, 6.73637754e-01,
        2.61710973e-02, 2.33698114e-02, 4.96201156e-01, 7.26618652e-01,
        4.77688590e-01, 3.71149192e-01, 4.38555491e-01, 1.30171188e-01,
        7.74385481e-01, 4.31251722e-01, 3.26659202e-01, 5.91690453e-02,
        6.55620336e-01, 3.33766179e-01, 8.89307068e-01, 2.86486707e-01,
        5.46990098e-01, 3.37959890e-01, 7.01678169e-02, 2.83089679e-01,
        3.64936434e-01, 5.52377353e-01, 7.34626564e-01, 5.26967261e-01],
       [6.46753687e-01, 4.21268495e-01, 8.77849391e-01, 9.49851436e-01,
        2.33430530e-01, 5.48942398e-03, 3.98083620e-01, 4.40970444e-01,
        9.38044645e-01, 1.03109876e-01, 2.22832674e-02, 4.16854515e-01,
        1.51023474e-01, 8.07808615e-01, 2.75967273e-01, 6.07606381e-01,
        9.17726306e-01, 9.35616140e-01, 9.20871487e-01, 2.06420214e-01,
        4.57876706e-01, 6.28021643e-01, 7.67145912e-01, 8.72261321e-01,
        3.22385788e-01, 6.54894487e-01, 6.89825688e-03, 4.17380824e-02,
        1.89910877e-01, 9.80832773e-01, 7.41254037e-01, 9.75081195e-01,
        1.26965233e-02, 2.55722323e-01, 5.89350015e-01, 6.76425326e-01,
        5.64016584e-01, 6.16096831e-01, 2.19915266e-01, 5.68322359e-01,
        6.67820887e-02, 4.18313448e-01, 4.78764060e-01, 9.86877246e-01],
       [4.40900258e-01, 7.32801550e-01, 6.15422854e-01, 1.94350802e-01,
        7.10745336e-02, 5.31835529e-01, 2.02451817e-01, 7.60552669e-01,
        2.18506234e-01, 4.58127636e-01, 9.16351147e-01, 7.22673416e-01,
        8.33694640e-01, 9.07747264e-01, 4.56834381e-01, 6.38902709e-01,
        2.57349590e-01, 6.36719290e-01, 6.54247189e-01, 7.53258256e-01,
        1.35730472e-01, 2.54828045e-02, 6.21279156e-01, 9.65069782e-01,
        3.71713666e-01, 4.24253169e-02, 4.34161865e-01, 5.17499215e-02,
        1.35104798e-01, 2.69574127e-02, 7.99215645e-01, 1.82340576e-01,
        4.55624155e-01, 6.24665615e-01, 5.63914128e-02, 8.41309150e-02,
        3.17676957e-01, 3.83408011e-01, 3.58764490e-01, 1.83955759e-01,
        2.60218234e-01, 3.76184529e-01, 2.84143544e-01, 9.81129055e-01],
       [3.13567501e-01, 3.07787851e-01, 7.35088452e-01, 4.04902356e-01,
        1.93589727e-02, 3.11570621e-01, 8.64845539e-01, 5.25781842e-01,
        6.37351583e-01, 1.16839697e-01, 4.53853140e-01, 4.52792691e-01,
        8.46629716e-01, 1.63177838e-01, 6.86570106e-01, 9.12269423e-01,
        1.55095812e-01, 7.20645073e-01, 9.00760424e-01, 5.54640712e-01,
        4.03705572e-01, 7.87637885e-01, 5.01210624e-01, 8.21848605e-01,
        5.57602726e-01, 6.22075067e-01, 4.36080961e-01, 9.24045019e-01,
        8.23440585e-01, 7.86331623e-01, 6.54038574e-01, 4.75575756e-01,
        3.52678282e-01, 3.55665481e-01, 2.68889882e-02, 6.00351686e-01,
        5.13108044e-01, 8.81438404e-01, 9.22591515e-01, 4.95178701e-01,
        4.13121741e-01, 9.14457691e-01, 8.19729747e-01, 5.47977108e-01],
       [1.14693820e-01, 2.71770275e-01, 9.17731497e-02, 7.55552913e-01,
        1.91784982e-01, 5.95952491e-02, 4.10818665e-01, 3.46170780e-02,
        8.44294368e-04, 5.47645026e-01, 9.25026133e-01, 6.19417492e-01,
        4.38648338e-01, 8.52119851e-01, 9.99535490e-01, 6.02851672e-01,
        5.48553106e-02, 7.80182026e-01, 4.87841449e-01, 2.33604881e-01,
        3.09707383e-01, 5.47535098e-01, 2.15153695e-01, 6.52717974e-01,
        2.39967052e-01, 1.29076119e-01, 7.79795374e-01, 2.48207942e-01,
        1.92188986e-01, 1.42544547e-01, 2.21199412e-01, 3.15942732e-01,
        3.95936645e-01, 6.12877783e-01, 4.30975892e-01, 7.69235329e-01,
        7.67841097e-01, 5.41959765e-01, 2.16471881e-01, 9.08632092e-01,
        1.60198696e-01, 5.17175754e-01, 3.43764506e-01, 4.58965340e-03],
       [5.42162852e-01, 5.82167363e-01, 2.68923291e-01, 1.31492471e-01,
        6.91019660e-01, 4.71194757e-02, 5.40809786e-01, 5.78429234e-01,
        3.44586013e-01, 9.49022535e-01, 6.50625011e-01, 1.85685475e-01,
        6.65818729e-01, 3.89384060e-01, 3.25906890e-01, 1.56202543e-01,
        1.96392350e-01, 1.92240025e-01, 9.46119240e-01, 4.84604845e-01,
        5.65324713e-02, 4.61215759e-01, 7.94206787e-01, 7.62547841e-01,
        6.61720295e-01, 8.42438414e-01, 2.95242043e-01, 1.87875982e-01,
        2.74648533e-01, 3.47932039e-01, 2.71694020e-01, 7.51257754e-01,
        2.07108857e-01, 9.83922586e-01, 7.49317281e-01, 8.72750620e-01,
        6.47085537e-01, 7.00355327e-01, 7.55623763e-01, 2.90689476e-01,
        1.84154527e-02, 6.29501988e-01, 9.40785919e-01, 2.54525635e-01],
       [2.92461655e-01, 5.43681316e-01, 6.67666495e-01, 6.50719849e-01,
        2.39138829e-01, 5.54971025e-01, 5.89705407e-01, 4.13332104e-01,
        9.51233519e-01, 3.80454044e-01, 6.79845349e-01, 9.86978318e-01,
        4.10255347e-02, 4.25011721e-01, 9.38699209e-01, 8.49814177e-01,
        7.51143696e-01, 9.12531342e-01, 5.88020457e-01, 1.96031390e-01,
        3.23784633e-01, 7.27116256e-01, 2.66093211e-01, 3.05228974e-01,
        7.94694994e-01, 1.79869573e-01, 1.55645062e-01, 8.72586726e-02,
        4.20241920e-01, 9.78674254e-01, 4.86530418e-01, 8.66855761e-01,
        4.00251214e-01, 3.86521898e-01, 1.70652525e-01, 2.46974530e-01,
        7.56095269e-01, 2.61136014e-01, 9.91494648e-01, 9.30539512e-01,
        6.06804896e-01, 1.18821295e-01, 3.88068165e-01, 2.91606251e-01],
       [1.07260249e-01, 9.50476165e-01, 8.04836673e-01, 7.50205035e-01,
        3.76420323e-01, 5.53193602e-01, 6.35771249e-01, 8.79537220e-01,
        5.30333481e-01, 5.87544225e-01, 8.60247349e-01, 7.18181527e-01,
        9.88017819e-01, 7.42098373e-01, 1.43744700e-01, 6.09139594e-01,
        9.27156533e-01, 3.19638852e-01, 9.91057644e-01, 2.39489822e-01,
        3.34731891e-01, 9.77988550e-01, 4.86019475e-01, 4.96038330e-01,
        6.80368473e-01, 6.19116360e-01, 1.57516337e-01, 6.84088852e-01,
        3.26520041e-01, 2.22313813e-01, 8.65191436e-02, 3.22688137e-01,
        8.72330593e-01, 9.47098756e-01, 1.24705009e-01, 7.14840264e-01,
        3.93128104e-01, 5.96789082e-01, 8.29442390e-01, 7.29384725e-01,
        2.69928513e-01, 4.59299002e-01, 8.58434125e-01, 7.01217289e-01],
       [1.78855694e-01, 3.13435862e-01, 7.92768469e-01, 9.88794997e-01,
        8.58356915e-01, 3.19914399e-01, 9.02771953e-01, 5.89532194e-01,
        6.56810596e-01, 3.66877737e-01, 1.26292252e-01, 9.59402183e-01,
        6.61501887e-02, 5.33424668e-01, 1.99868685e-01, 6.35532598e-01,
        6.76735082e-01, 6.84047796e-01, 6.03814904e-01, 9.85810643e-01,
        5.07249935e-01, 1.38837521e-01, 6.96380973e-01, 4.52063429e-01,
        2.40296646e-01, 4.76851015e-01, 1.15388231e-01, 3.83429894e-01,
        6.63508758e-01, 1.78167479e-02, 8.58757901e-01, 1.42289328e-01,
        6.51486423e-02, 4.53761029e-01, 3.56834371e-01, 9.24483806e-01,
        7.49343752e-01, 5.14320089e-01, 1.54070000e-04, 6.31840180e-01,
        7.58437278e-01, 7.46422670e-01, 4.88538813e-01, 2.21031191e-02],
       [2.49496454e-01, 7.82412893e-02, 4.15047687e-01, 4.35569476e-01,
        4.88826866e-01, 6.22154333e-01, 8.81181099e-04, 5.79476947e-01,
        6.08973824e-01, 1.64875561e-01, 5.50902296e-01, 4.82429814e-01,
        6.52327059e-01, 5.38967289e-01, 9.56346404e-01, 6.60515175e-01,
        6.42072414e-01, 9.83154959e-01, 1.38261608e-01, 7.80011637e-01,
        9.49786030e-01, 4.68319199e-01, 8.99467666e-01, 8.00568315e-02,
        2.90045417e-01, 4.84625546e-01, 7.94033870e-01, 8.78770917e-01,
        7.58244432e-01, 9.20662782e-01, 5.17938875e-01, 8.39920295e-01,
        3.30878409e-01, 1.81132353e-01, 7.45391660e-02, 6.52996603e-01,
        2.34917571e-01, 7.41350824e-01, 2.32671115e-01, 2.89512088e-01,
        5.47666899e-01, 9.97408873e-01, 6.95335025e-01, 5.43250149e-01],
       [2.33834354e-01, 2.13252617e-01, 6.54965605e-01, 4.49310670e-01,
        3.42253852e-01, 7.28313921e-01, 6.47439001e-01, 5.03720692e-01,
        8.04791389e-02, 9.40575054e-01, 4.08702254e-01, 3.67740444e-01,
        1.91508460e-01, 4.48331010e-01, 1.77790895e-01, 7.74416605e-01,
        6.40744874e-01, 3.59189898e-01, 5.34796200e-01, 6.73015439e-01,
        9.16573461e-01, 9.47026137e-01, 7.70025864e-01, 5.53745251e-01,
        8.99346825e-01, 9.79651198e-01, 9.16272796e-02, 6.80282396e-01,
        3.43743354e-01, 4.88192261e-01, 8.94250180e-01, 6.02052485e-01,
        7.90852417e-02, 1.74642226e-01, 6.22291698e-01, 6.41005762e-01,
        4.66674098e-02, 8.50083612e-01, 3.19151970e-01, 2.27465168e-01,
        1.57859940e-01, 4.35569333e-01, 1.25779417e-01, 4.36352789e-01],
       [7.23591158e-01, 2.64543186e-02, 2.70717190e-01, 1.04369962e-01,
        2.82346914e-02, 6.54460821e-01, 9.40743913e-01, 5.67654568e-02,
        4.05950138e-01, 3.12931868e-01, 8.44398473e-01, 1.11998625e-01,
        7.26203814e-01, 2.23574031e-01, 6.13785399e-01, 4.51208423e-01,
        5.93687045e-01, 7.76698869e-01, 9.94505727e-01, 1.74368525e-02,
        6.05514107e-02, 6.16793034e-01, 3.00326434e-01, 3.57946758e-01,
        8.99176984e-01, 6.12322205e-01, 5.52067123e-01, 7.67699597e-01,
        3.78989503e-02, 8.77238978e-01, 4.94560590e-01, 7.81247057e-01,
        2.44766558e-01, 6.79743841e-01, 2.82340227e-01, 8.64979035e-01,
        8.34050414e-01, 2.83283453e-01, 2.10118592e-01, 6.57021471e-01,
        7.20692835e-01, 2.40632956e-01, 3.62991272e-01, 3.76307768e-01],
       [7.95772842e-01, 2.42724981e-01, 8.13341203e-01, 6.68972562e-01,
        5.30760468e-01, 3.80771056e-01, 2.05791049e-01, 4.48947189e-01,
        8.76058941e-01, 8.31668519e-01, 7.94083246e-02, 2.98807096e-01,
        8.71079269e-01, 6.31270868e-01, 5.94339870e-01, 7.26295301e-01,
        6.29507922e-01, 1.82261317e-01, 2.75577928e-01, 9.28287354e-01,
        6.84988615e-01, 3.55959057e-01, 9.85692975e-01, 3.97657377e-01,
        8.80989799e-01, 1.89725053e-01, 9.32229943e-01, 6.08951421e-01,
        9.24978699e-01, 3.73943878e-01, 2.58277540e-01, 2.57393847e-01,
        4.31803472e-01, 9.08224594e-01, 2.52230124e-01, 5.61971596e-01,
        3.49178188e-01, 6.30762084e-01, 1.87815622e-01, 6.26627114e-01,
        9.06629619e-01, 4.28746089e-01, 1.27310468e-01, 5.99794339e-01],
       [2.51060416e-01, 3.46085110e-02, 4.84991217e-02, 5.30603269e-01,
        6.79144370e-01, 6.76332831e-02, 9.42401789e-01, 2.17303753e-01,
        6.31160844e-01, 4.68086760e-01, 6.05692680e-01, 1.41949607e-01,
        7.23483433e-01, 8.25391260e-01, 8.34347136e-01, 2.57072564e-01,
        3.57760511e-02, 5.07437587e-01, 2.12702036e-01, 9.49298781e-01,
        1.63978885e-01, 7.49853668e-01, 3.52005195e-01, 9.12179057e-01,
        1.63582724e-01, 6.61922487e-01, 7.27277055e-01, 2.10539323e-01,
        8.41555575e-01, 1.12283526e-01, 9.81428560e-01, 7.69289391e-02,
        5.75457600e-01, 6.48631982e-02, 9.55173641e-01, 5.15713835e-01,
        4.18181421e-01, 6.71116241e-01, 3.88788075e-01, 7.15623730e-01,
        2.40997346e-01, 1.18429423e-01, 6.01420042e-01, 2.51900887e-01],
       [8.83078305e-01, 8.05659165e-01, 7.75219183e-01, 2.39648032e-01,
        8.42976997e-01, 2.53118267e-01, 7.44993951e-01, 1.22040029e-01,
        5.39903747e-01, 8.23857194e-01, 9.45688506e-01, 9.10321488e-01,
        4.80693629e-01, 6.20133652e-02, 1.55773238e-01, 9.85789541e-01,
        4.59272794e-01, 5.18757032e-01, 6.74480042e-01, 3.63261885e-01,
        6.20403010e-01, 8.74131668e-01, 4.67605153e-02, 5.49182798e-01,
        7.72623622e-01, 2.52783676e-01, 2.22318140e-01, 7.35728648e-01,
        3.23486163e-01, 8.66852291e-01, 6.13762843e-01, 8.04018259e-01,
        7.76950083e-01, 7.57065754e-01, 7.24563415e-02, 5.36725586e-01,
        3.79342147e-01, 5.14103031e-01, 3.96980673e-01, 1.59954526e-01,
        8.75136464e-01, 7.87786756e-02, 3.09934744e-01, 2.08338910e-01],
       [9.18534706e-01, 7.77408958e-01, 5.65646507e-01, 5.50051742e-01,
        5.63623210e-01, 3.45803798e-01, 9.09144925e-01, 6.25469962e-01,
        7.15859652e-01, 6.39093263e-02, 9.54877977e-01, 1.62509002e-01,
        5.92840562e-01, 7.28265973e-02, 6.43158249e-01, 4.77942134e-01,
        6.02406020e-01, 8.69202562e-01, 4.58196243e-01, 7.56661008e-01,
        3.85547999e-02, 5.34704837e-01, 4.09573396e-01, 4.22403427e-01,
        3.59908291e-01, 3.09413954e-01, 1.97775336e-01, 2.51203764e-01,
        6.60086701e-02, 9.59629196e-01, 5.51023304e-01, 6.13436491e-01,
        9.99400289e-01, 8.37269564e-01, 8.89594033e-01, 1.47158263e-01,
        4.53094885e-01, 1.01137751e-02, 6.22393809e-01, 1.87827758e-01,
        1.71722311e-04, 5.07293591e-01, 1.10733733e-01, 2.75522490e-01],
       [8.28777227e-01, 6.50059415e-02, 5.53645294e-01, 2.24708090e-01,
        1.09253385e-01, 1.43113673e-01, 5.92652424e-01, 6.50794227e-01,
        6.27236003e-03, 9.46539295e-02, 5.80439451e-01, 6.85773906e-01,
        5.26377090e-01, 1.52492685e-01, 6.27441958e-02, 4.10661360e-01,
        8.80871833e-01, 8.87310865e-01, 7.38931297e-02, 6.44007973e-01,
        3.27074930e-01, 3.78419949e-01, 9.38881977e-01, 1.58875575e-01,
        5.51535627e-01, 4.06553703e-02, 7.48281949e-01, 4.52667869e-01,
        3.48851082e-01, 9.81037439e-01, 5.85082629e-01, 9.52376688e-02,
        1.43059436e-01, 1.31712920e-01, 5.34137420e-01, 5.22828621e-03,
        7.92630440e-01, 6.77595628e-01, 4.95990790e-01, 3.91123297e-01,
        5.63300198e-01, 4.91788407e-01, 4.79061516e-01, 1.33361441e-01],
       [4.10599597e-01, 5.69360618e-01, 6.64077069e-01, 2.26819791e-01,
        2.12393102e-01, 9.33699221e-01, 4.12416898e-01, 8.57740487e-01,
        5.98161623e-01, 5.99443235e-01, 8.18545686e-01, 3.74092082e-01,
        1.67186519e-01, 4.54003699e-02, 6.87633040e-01, 1.81790694e-01,
        5.37429187e-01, 8.41828730e-01, 9.22419000e-01, 6.97780011e-01,
        8.31671162e-01, 6.45652476e-01, 8.48987119e-01, 2.34104607e-01,
        2.04160707e-01, 4.19072628e-01, 6.04656695e-01, 5.13944919e-01,
        6.06021427e-01, 8.63701928e-01, 1.11276870e-01, 9.00529087e-01,
        8.55471780e-01, 6.33989128e-01, 4.47880802e-01, 3.68174925e-03,
        5.59970673e-01, 3.75078859e-01, 2.61151572e-01, 7.33987195e-01,
        8.11272123e-01, 6.18748874e-01, 5.50559125e-01, 2.57089887e-01],
       [8.47017864e-01, 8.92723000e-01, 1.22231514e-01, 2.41837647e-01,
        8.20333179e-01, 6.95200441e-01, 2.20821182e-01, 6.51996301e-03,
        9.70975924e-02, 2.78635444e-01, 4.64787595e-02, 4.22556792e-01,
        6.63641208e-01, 7.53877657e-01, 1.43600494e-01, 8.13774475e-01,
        1.69060709e-01, 3.04098937e-01, 9.72100805e-02, 3.06290214e-01,
        8.15520857e-01, 3.31617168e-02, 5.71951630e-01, 6.57356588e-01,
        3.74493387e-01, 8.56152355e-02, 4.78594843e-01, 7.57871392e-01,
        9.78792934e-01, 4.62197411e-01, 6.90465940e-01, 9.34431149e-01,
        4.71343848e-01, 3.64293959e-01, 5.24859595e-01, 2.77058921e-01,
        5.50968497e-01, 6.66509524e-01, 4.95822861e-01, 6.00363804e-01,
        9.70452831e-01, 6.26736743e-01, 3.47077668e-01, 7.31518801e-01],
       [7.60081286e-01, 2.55133447e-01, 4.86787045e-02, 7.74258492e-01,
        7.42004010e-01, 3.06055442e-01, 9.51650280e-01, 6.09014512e-01,
        8.40394505e-01, 4.74149832e-01, 7.36040942e-01, 9.27798600e-01,
        8.38015174e-01, 9.59492490e-02, 8.45193786e-01, 3.02050692e-01,
        4.87859879e-01, 1.78981986e-01, 1.18968402e-01, 8.76031412e-01,
        8.53019829e-01, 5.99053315e-01, 2.54809653e-01, 7.50501777e-01,
        2.27764507e-01, 2.23993968e-01, 8.40684600e-01, 9.39980797e-01,
        4.91488326e-01, 7.64043583e-02, 9.20863982e-01, 8.01781927e-01,
        7.82914644e-01, 2.42494918e-02, 4.98267635e-01, 3.30703315e-01,
        3.77414567e-01, 4.68480159e-01, 3.11813957e-01, 8.72217836e-01,
        5.52777295e-01, 2.38912719e-01, 3.49221406e-01, 6.85397984e-01],
       [4.00633809e-01, 6.07896255e-01, 4.39089384e-01, 3.21493605e-01,
        6.65729302e-01, 6.47298419e-01, 7.80763495e-02, 5.07265835e-01,
        5.63510230e-01, 8.45843170e-01, 6.95215080e-01, 9.44599750e-01,
        8.50785033e-01, 4.09923396e-01, 3.45058230e-01, 4.49656504e-01,
        1.35839022e-01, 6.14520183e-01, 8.48773186e-01, 6.11783457e-01,
        6.48346708e-01, 9.54950510e-01, 8.51376473e-01, 6.58610755e-01,
        5.53183742e-01, 5.86445145e-01, 4.63412191e-01, 2.96178458e-01,
        4.39483708e-01, 6.88659877e-01, 8.48175196e-01, 1.69088638e-01,
        1.45490670e-01, 4.35149208e-01, 8.93577132e-01, 7.89587714e-01,
        3.87656273e-01, 3.19441506e-01, 8.45320534e-01, 8.16879873e-01,
        9.23067393e-02, 9.99992805e-01, 4.21411126e-01, 2.95121765e-01],
       [2.29160097e-01, 7.20218943e-01, 3.56397152e-01, 2.80958835e-01,
        7.33130397e-01, 9.10767431e-02, 3.30662715e-02, 3.12470226e-01,
        7.11930440e-02, 3.96613806e-01, 1.90280337e-01, 2.80713941e-01,
        4.10900445e-01, 9.95071132e-02, 2.24057892e-01, 3.55474696e-01,
        3.25397101e-01, 7.64811863e-02, 9.89545031e-01, 7.33932734e-01,
        9.43769554e-01, 3.39035110e-01, 2.37936272e-03, 9.59919528e-01,
        9.05489935e-01, 7.14937031e-03, 2.83616505e-01, 9.80113221e-01,
        4.47623964e-01, 1.32355094e-01, 4.40064707e-01, 9.73736243e-02,
        5.38733957e-02, 3.03221408e-01, 2.29554891e-01, 5.35866030e-01,
        8.12761833e-02, 4.62147365e-01, 4.56417160e-01, 5.89621024e-01,
        1.25567731e-01, 5.75215264e-01, 1.40332071e-01, 9.87073098e-01],
       [9.73551981e-01, 6.71827359e-01, 3.05795247e-01, 1.08982881e-01,
        5.07061489e-01, 8.19652044e-01, 2.31175044e-01, 5.96872521e-01,
        3.80042644e-02, 7.79175678e-01, 1.64229524e-01, 5.52248764e-02,
        4.95848189e-01, 3.18655823e-01, 7.41165385e-01, 4.50879585e-01,
        4.73896151e-01, 4.90339551e-01, 8.67184869e-01, 5.46959857e-01,
        3.78954895e-01, 8.82244790e-01, 4.78433125e-01, 2.86738202e-01,
        9.55920296e-01, 8.07255570e-01, 3.27499861e-01, 3.41923004e-01,
        2.09948672e-01, 8.19298325e-02, 8.56909662e-01, 3.72693883e-02,
        1.86774532e-01, 8.00022911e-01, 1.98353771e-01, 3.66557799e-01,
        3.82904217e-01, 2.71230658e-01, 7.49536178e-01, 1.58857627e-01,
        3.37880523e-01, 9.61546631e-01, 9.00478115e-03, 2.96140834e-01],
       [5.70420518e-01, 3.16240042e-01, 6.42750154e-01, 8.65877565e-01,
        3.51994518e-01, 5.20279738e-01, 7.60546367e-01, 9.00196819e-01,
        5.44679076e-01, 1.32948523e-01, 9.82356655e-01, 4.54503955e-01,
        9.40682132e-01, 9.26630101e-02, 4.09518366e-01, 1.65962881e-01,
        7.93797825e-01, 2.57582416e-01, 4.34004896e-02, 5.42470557e-01,
        3.82731567e-01, 9.48625937e-01, 5.85516358e-01, 3.32037998e-01,
        4.74272980e-01, 2.62779654e-01, 8.70736085e-01, 6.00703792e-02,
        3.55525886e-01, 6.48231684e-01, 7.94141987e-01, 6.29107346e-01,
        2.16600640e-01, 7.54817720e-01, 3.18826938e-01, 9.67035758e-01,
        4.71502969e-01, 7.41048244e-01, 9.74974113e-01, 1.34808812e-02,
        5.08504043e-01, 2.52275724e-01, 8.22575725e-01, 9.59070940e-01],
       [2.30730918e-01, 9.51294918e-01, 8.40945229e-01, 7.32558068e-01,
        4.49252366e-01, 2.50282668e-01, 8.63785322e-01, 9.99811035e-01,
        1.49571964e-01, 3.40767385e-01, 7.22870666e-02, 2.48805647e-01,
        6.85554488e-01, 2.05091224e-01, 3.49322084e-01, 5.36451175e-01,
        7.43506610e-01, 3.44611315e-01, 4.55211361e-02, 9.54242396e-02,
        8.91920921e-01, 5.98315556e-01, 5.03084053e-01, 8.94729227e-01,
        6.87634713e-01, 9.17405507e-01, 7.30302269e-02, 9.99640689e-01,
        2.40779329e-01, 8.67737307e-01, 5.74332338e-01, 6.93528582e-01,
        1.68094388e-01, 7.24168999e-01, 4.95182133e-01, 1.08803747e-01,
        9.32604989e-02, 4.55582973e-02, 9.35295730e-01, 8.04929821e-01,
        4.14827489e-01, 8.89075713e-01, 3.73287993e-01, 3.68784333e-01],
       [7.67365306e-02, 3.71931205e-01, 6.31949462e-01, 8.95612995e-01,
        7.23976898e-01, 9.13397592e-01, 9.75319804e-01, 6.32618627e-01,
        1.74095359e-01, 3.77958169e-01, 6.64409609e-01, 5.92188250e-01,
        3.90237480e-01, 7.24057644e-01, 6.86678857e-01, 2.68143874e-02,
        5.37911159e-02, 2.33069702e-01, 9.10652388e-02, 7.06681576e-01,
        2.63949109e-01, 9.36815020e-01, 9.04835865e-01, 4.46367531e-01,
        8.03608949e-01, 5.47848349e-01, 6.51104875e-02, 4.33237845e-01,
        5.98704175e-01, 2.83599277e-01, 1.47339658e-02, 7.01151315e-02,
        8.81847456e-01, 1.25678373e-01, 7.87029672e-01, 3.82968895e-01,
        4.49658489e-01, 9.80277568e-01, 5.18420262e-01, 3.27686155e-01,
        2.38575070e-01, 7.88982257e-01, 9.96783164e-01, 2.04512179e-01],
       [9.19686368e-01, 6.95395939e-01, 8.62096976e-01, 5.97309160e-01,
        7.81056795e-01, 5.76958383e-01, 4.79968151e-01, 5.55774036e-01,
        6.07320390e-01, 2.74093850e-01, 7.40939327e-02, 7.78786493e-01,
        9.78388873e-01, 4.73308024e-01, 3.72619841e-01, 6.38802407e-01,
        8.44517649e-01, 3.61040366e-01, 5.64255005e-01, 3.02509373e-01,
        6.89786869e-01, 1.64539268e-01, 3.63253358e-01, 5.22301199e-01,
        5.88782017e-02, 8.71048096e-01, 3.98848946e-02, 6.70286719e-01,
        9.17731691e-01, 1.23173816e-02, 4.47388499e-01, 4.93188950e-01,
        9.35902041e-01, 8.95485010e-01, 1.64158599e-01, 2.03330894e-01,
        6.52462381e-01, 9.45056504e-01, 6.08046839e-01, 1.71348737e-01,
        9.76312583e-01, 9.11857133e-01, 5.00751609e-01, 8.62789661e-01],
       [6.09578930e-01, 8.80962573e-01, 8.86235760e-01, 4.79630974e-01,
        7.63107333e-01, 6.98150544e-01, 2.73073488e-01, 8.37943361e-01,
        4.29107869e-01, 6.83157647e-01, 2.09765122e-01, 6.56170179e-01,
        1.05060932e-01, 5.60489524e-01, 2.40173759e-01, 8.55040595e-01,
        4.95563996e-01, 3.59604599e-01, 2.80990674e-01, 8.12653551e-01,
        7.43758723e-01, 8.54742460e-01, 7.33971109e-02, 4.69065855e-01,
        1.38568234e-01, 1.72066239e-01, 5.60832998e-01, 7.04933012e-01,
        9.77810391e-01, 5.32690212e-01, 2.59506639e-01, 3.95551878e-01,
        3.36617082e-01, 1.20529013e-01, 2.33618233e-01, 8.33779982e-01,
        7.01148997e-01, 9.09734436e-01, 7.00164101e-01, 5.88948332e-01,
        4.85348381e-02, 6.67529237e-01, 5.97907403e-01, 9.97768901e-01],
       [2.52286166e-01, 5.36057385e-02, 7.24000490e-01, 3.20897489e-01,
        4.04412197e-01, 9.49344554e-01, 4.90418798e-01, 6.71459966e-02,
        9.55914442e-01, 3.34557194e-01, 6.19503691e-02, 2.54892061e-01,
        8.15403532e-01, 9.50726498e-01, 2.23655114e-01, 6.95761901e-03,
        9.85477573e-02, 9.01643223e-01, 6.32417635e-01, 2.48974493e-01,
        9.31407253e-02, 3.52079091e-01, 7.43844454e-01, 1.73081127e-01,
        3.77828757e-01, 5.18979359e-01, 3.20382974e-01, 6.03231806e-01,
        8.66270349e-01, 5.85551750e-01, 4.18351252e-02, 6.06456576e-01,
        2.73141388e-01, 6.58525748e-01, 5.65613827e-01, 3.32437218e-03,
        7.31937982e-01, 5.73658417e-01, 9.70654061e-02, 4.46612558e-01,
        5.84383013e-01, 2.05411961e-01, 8.46981590e-01, 9.90241904e-01],
       [8.93582761e-01, 8.49270960e-01, 1.01516952e-03, 7.43174509e-01,
        9.49175496e-01, 2.40451841e-01, 1.74251098e-03, 7.19251579e-01,
        1.26189528e-01, 7.86464244e-01, 6.08108273e-01, 7.08997774e-01,
        1.37678541e-01, 5.33560263e-01, 4.53065156e-01, 5.58781022e-01,
        9.58315810e-01, 4.99517408e-01, 4.36655085e-01, 8.29466223e-01,
        2.55027143e-01, 1.01508397e-01, 9.14238902e-01, 9.36460796e-01,
        7.83978936e-01, 2.43484580e-01, 3.55680256e-01, 3.21204795e-01,
        6.83287645e-01, 9.93404093e-01, 2.80643250e-01, 6.60591068e-01,
        6.40730349e-01, 9.82416193e-01, 9.38729975e-01, 5.90136783e-01,
        4.02042087e-01, 4.85193983e-01, 4.53722955e-01, 4.38642208e-01,
        5.01445720e-01, 6.60941748e-01, 8.35840352e-01, 8.30299798e-01],
       [5.89293562e-01, 9.18148832e-01, 2.01798724e-01, 3.58203728e-01,
        7.97391054e-01, 8.68674228e-01, 4.63688080e-01, 4.33512042e-01,
        2.40096606e-01, 8.83850607e-01, 6.06169149e-01, 7.93384874e-01,
        2.92031541e-01, 5.68779904e-01, 1.60094452e-01, 6.60963567e-01,
        5.81649301e-01, 3.64509188e-01, 4.11894842e-01, 9.12018741e-01,
        3.16379918e-01, 7.70831418e-01, 4.75086858e-01, 8.96640852e-01,
        3.68035454e-01, 9.94350887e-01, 5.82432427e-02, 1.44319924e-01,
        3.96944250e-01, 5.26767229e-01, 4.53424640e-01, 6.54465747e-01,
        7.01200457e-01, 5.66885678e-01, 1.84062672e-01, 7.67011842e-01,
        8.33612444e-02, 5.97439214e-01, 2.99965843e-01, 7.06897319e-01,
        8.13827190e-01, 6.50552244e-02, 3.49923793e-01, 4.94756984e-01],
       [4.79220585e-01, 4.18325916e-02, 4.44234261e-01, 3.14031195e-01,
        4.40791504e-01, 7.30409590e-01, 1.54886138e-01, 8.01785000e-01,
        1.05242461e-01, 5.83424177e-01, 8.67483642e-01, 6.69309167e-01,
        7.76476275e-01, 8.87983041e-01, 6.12824081e-01, 1.83432850e-01,
        1.54309648e-01, 7.95545932e-01, 8.24165708e-01, 3.16821403e-01,
        1.38956216e-01, 6.55961617e-01, 5.33396812e-01, 4.77991094e-01,
        3.80628746e-01, 4.05353919e-01, 4.35406267e-02, 5.05192213e-01,
        3.08108349e-01, 6.08738429e-01, 3.60738976e-01, 6.69634692e-01,
        6.48385017e-01, 5.98575374e-02, 7.98301928e-01, 6.44254918e-01,
        4.42093003e-01, 8.70572408e-01, 9.98963512e-01, 9.88762086e-01,
        7.46521924e-01, 2.38599076e-01, 4.29674937e-01, 8.82324601e-01],
       [4.89399800e-01, 2.01952134e-01, 1.49776054e-01, 5.83418289e-01,
        3.90411706e-01, 7.79766571e-01, 5.80766913e-01, 5.91721999e-01,
        1.91255976e-01, 4.65004683e-01, 3.59665653e-01, 5.70794447e-01,
        5.31662635e-01, 1.55266814e-01, 1.89110950e-01, 2.30390169e-01,
        4.47206953e-01, 3.12268777e-02, 5.65489075e-01, 9.35942022e-01,
        1.24951437e-01, 9.55550561e-01, 9.21070463e-01, 9.83615959e-01,
        7.61562031e-01, 2.06967379e-02, 7.35212027e-01, 1.32768804e-01,
        8.51424890e-01, 8.35962164e-01, 8.40348328e-04, 9.72850560e-01,
        3.82780268e-01, 9.85932668e-01, 6.15441217e-01, 7.50745267e-01,
        4.35729523e-01, 8.23118750e-01, 8.80698096e-01, 3.08782675e-01,
        9.19233260e-01, 6.19751373e-01, 9.40231477e-01, 5.64255803e-01],
       [9.79367879e-01, 7.62874075e-01, 9.22315041e-01, 6.39345165e-01,
        7.49309938e-01, 6.18117994e-01, 6.77679204e-01, 3.60931291e-01,
        7.43040800e-01, 6.68228621e-01, 5.84479551e-02, 7.95318335e-02,
        6.53989554e-01, 8.75688996e-01, 6.37475909e-01, 1.38648750e-01,
        9.45665342e-01, 4.76740596e-01, 8.83589762e-01, 1.82623316e-01,
        9.52238022e-01, 3.50107005e-01, 7.89830592e-01, 3.84139377e-01,
        4.91116426e-01, 9.47360200e-01, 1.29229547e-01, 8.11261247e-01,
        4.59288375e-01, 9.84127159e-01, 7.18732333e-01, 6.48044032e-01,
        4.81343076e-01, 5.70337978e-01, 2.07556257e-01, 9.38636529e-01,
        1.57653970e-01, 7.64545306e-01, 7.21279870e-02, 8.95049463e-01,
        2.48921660e-01, 3.56611274e-01, 9.50097485e-01, 6.99299771e-01],
       [3.89319477e-01, 7.16529770e-01, 9.59292197e-01, 2.81073781e-01,
        5.56054475e-01, 3.76644767e-01, 3.56430340e-01, 3.11145297e-02,
        9.08886919e-01, 4.82511205e-01, 5.50009450e-01, 1.14716891e-01,
        2.39079640e-01, 9.95769445e-01, 4.91321217e-01, 9.78470555e-01,
        3.64381724e-02, 6.30489454e-01, 8.28014206e-01, 1.75849101e-01,
        6.75995129e-01, 4.80108318e-01, 3.20104276e-02, 2.68601225e-01,
        2.48234321e-01, 6.19354951e-01, 5.68194453e-02, 9.43197227e-01,
        3.86597470e-01, 2.31050379e-01, 5.44961645e-01, 3.61743759e-01,
        7.96649791e-01, 5.44252857e-01, 7.04963932e-01, 8.43332966e-01,
        2.63706533e-01, 2.95996083e-01, 7.04936298e-01, 6.62750887e-02,
        3.91460222e-01, 9.46931674e-01, 4.65441504e-02, 8.70687238e-01],
       [6.94472922e-01, 6.13443198e-02, 2.80288905e-01, 7.06795953e-02,
        8.43487107e-02, 1.20943711e-01, 5.25620082e-01, 1.04260189e-01,
        3.76979240e-01, 2.41698403e-01, 4.16760882e-01, 1.93434794e-01,
        8.49702599e-01, 1.00335114e-01, 6.44279905e-01, 2.01264303e-01,
        5.63272169e-01, 8.77963420e-01, 3.23299676e-01, 3.96194424e-01,
        6.14284885e-01, 2.54743526e-01, 4.16814740e-01, 5.55466595e-02,
        7.88021180e-01, 9.20097114e-01, 3.87483596e-02, 2.88566143e-01,
        9.83580227e-01, 3.33995320e-01, 8.04932327e-02, 5.96701513e-01,
        2.62674166e-01, 2.67929135e-01, 3.14488668e-01, 8.52871810e-01,
        4.47407774e-01, 9.38117995e-01, 7.59849616e-01, 7.08048661e-01,
        8.82176071e-01, 4.98574213e-02, 7.02885502e-01, 4.50105665e-01],
       [3.28372701e-01, 7.26948147e-01, 9.03585703e-01, 7.91956326e-01,
        4.88177106e-01, 5.23235657e-01, 5.04466356e-01, 4.43409246e-01,
        4.37834864e-01, 8.05049490e-02, 5.00498590e-02, 3.49304512e-01,
        2.09152674e-01, 5.09630740e-01, 2.88776688e-01, 8.99546006e-01,
        8.82691256e-01, 5.51010222e-01, 5.52193056e-01, 3.47506571e-02,
        9.89975134e-01, 5.77824911e-01, 9.73697654e-01, 9.47527874e-01,
        9.24622703e-01, 6.54693344e-01, 1.89861940e-01, 8.69984931e-01,
        9.46982436e-04, 5.27486785e-01, 6.13129191e-01, 6.84449197e-01,
        2.87049314e-01, 9.19518119e-01, 7.64016416e-01, 9.81922600e-01,
        3.11332498e-01, 9.62036755e-01, 9.16259129e-01, 2.71556044e-01,
        6.31862412e-01, 2.95980650e-01, 1.11780465e-01, 8.41253425e-01],
       [5.98453666e-01, 6.23674952e-01, 4.93871356e-01, 4.17925329e-01,
        5.97364746e-02, 2.65222716e-01, 1.05855777e-01, 5.80549449e-01,
        6.77963664e-01, 4.37633083e-01, 2.94495505e-01, 8.57557467e-01,
        2.99551116e-01, 9.12474905e-01, 3.66659392e-01, 1.54626885e-01,
        9.74701131e-01, 4.83418367e-02, 8.92257887e-01, 1.87085395e-01,
        1.59709597e-01, 9.55477324e-01, 1.75871506e-01, 5.54402235e-01,
        3.73132304e-01, 4.60956697e-01, 2.23934385e-01, 3.22245012e-01,
        3.70393466e-01, 8.86635625e-01, 1.79709203e-01, 1.06714308e-01,
        7.93344647e-01, 7.89805632e-01, 6.04041848e-01, 3.81044993e-01,
        9.52444742e-01, 4.79074580e-01, 2.88495513e-01, 9.91001600e-02,
        2.61575285e-01, 1.87146107e-01, 5.23619582e-01, 1.75670283e-01],
       [7.51529535e-01, 9.28467018e-01, 3.30505104e-01, 8.66504810e-01,
        2.64971941e-01, 7.15244823e-01, 8.82854459e-01, 6.27911116e-01,
        4.68858499e-01, 9.39190740e-01, 6.58289834e-01, 3.41147348e-01,
        6.09749305e-01, 7.26387020e-01, 6.59052475e-01, 5.49459763e-01,
        4.53255630e-01, 3.99619231e-02, 1.45039723e-01, 2.23666044e-01,
        2.09605536e-02, 1.37991302e-01, 4.25655508e-02, 9.76400275e-01,
        2.07389320e-01, 9.81482538e-01, 9.98114290e-01, 9.90389487e-01,
        3.35785823e-01, 1.88109629e-01, 5.80172523e-01, 6.18248347e-01,
        8.71172950e-01, 8.89501484e-01, 4.26718794e-01, 9.75509133e-01,
        7.05013527e-01, 1.53678845e-01, 3.12245772e-01, 9.93174948e-01,
        4.47206356e-01, 7.86793603e-01, 5.41365683e-02, 3.89780750e-01],
       [4.91766401e-01, 5.34960871e-01, 1.70359794e-01, 3.18330152e-01,
        9.27034331e-01, 7.98015058e-01, 8.96563367e-01, 6.86288502e-01,
        4.43868609e-02, 1.57608873e-01, 7.27438180e-01, 4.10283158e-01,
        4.10869999e-01, 1.68185772e-01, 9.98469211e-01, 2.73030839e-01,
        2.00721316e-01, 2.72125203e-01, 2.21541161e-01, 5.06618771e-01,
        5.16705716e-01, 7.94339024e-01, 4.57107238e-01, 8.75481199e-02,
        7.57637669e-01, 6.92163797e-01, 3.50379684e-01, 7.97985418e-01,
        6.45600481e-01, 8.18235496e-01, 7.58567282e-01, 8.60380439e-01,
        2.29038440e-01, 4.09327619e-01, 1.98262887e-01, 8.23705184e-01,
        8.24719095e-02, 4.01787157e-01, 4.53651348e-01, 6.27705829e-02,
        7.30875551e-02, 9.53562781e-01, 1.41880523e-01, 3.64260220e-01],
       [2.78967701e-01, 9.98174855e-01, 7.72232214e-01, 5.92804352e-01,
        2.00249994e-01, 3.00119167e-01, 5.77665681e-01, 1.91317212e-01,
        9.47062102e-01, 3.89786021e-01, 4.24299472e-03, 2.93692401e-01,
        7.30409903e-01, 5.22317288e-01, 8.60258368e-01, 4.38608993e-01,
        6.75606942e-01, 8.19890392e-01, 3.74073443e-02, 2.71688095e-01,
        6.07296254e-01, 6.20591540e-01, 4.31700927e-01, 7.06360614e-01,
        6.85309539e-01, 8.87793441e-01, 2.36181513e-01, 8.90348694e-01,
        7.88500792e-01, 7.02377298e-01, 1.00938845e-01, 2.71142046e-01,
        1.21312591e-01, 1.46245685e-01, 9.06237996e-01, 6.04295835e-01,
        3.21876004e-01, 4.93128255e-01, 1.43119451e-01, 8.77668809e-01,
        9.17894281e-01, 1.78265817e-01, 2.61046851e-01, 7.13783319e-01]])