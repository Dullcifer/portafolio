
#pwd
#conocer ruta donde nos encontramos

#ls
#listar archivos existentes en la ruta

#ls -la listar archivos existentes en la ruta

#cd carpeta
#change directory, cambiar directorio

#cd ..
#change directory, un paso atras

#git clone https://github.com/adrianedutecno/portafolio.git

#eliminar carpeta .git dentro de la carpeta portafolio descargada

#abrir proyecto dentro de visual studio code

#---

#1.* git status | ver status del repositorio
#2.* git config --global user.name "nombreUsuario"
#3.* git config --global user.email "tucorreo@gmail.com"
#4.* git config --list | listar configuración git
#5.* git init | inicializar repositorio

#abrir github.com y crear un repositorio

#* ![1756430155268](image/pasos_repo/1756430155268.png)

#6* git remote add origin https://github.com/{usuario}/{repo}.git
#7* git remote -v | visualizar repositorios remotos

#1.* ls -la ~/.ssh | consultar si existe una llave ssh en nuestro pc
#2.* ssh-keygen -t ed25519 -C "your_email@example.com" | crear una llave ssh
#3.* se nos pedira si queremos aceptar la ruta de creación, seleccionar ruta default
#4.* se nos pedira entrar una passphrase, se puede dejar vacío, o una frase sencilla
#5.* eval "$(ssh-agent -s)" | consultar por el agente ssh y activarlo
#6.* ssh-add ~/.ssh/id_ed25519 | añadimos una clave privada
#7.* cat ~/.ssh/id_ed25519.pub | clip   | leer la llave publica dentro de la terminal
#8.* ir a github, luego ingresar a settings del perfil e ir a ssh and gpg keys

#1.* git branch -M main
#2.* git add .
#3.* git commit -m "init commit"
#2.* git push -u origin main
 