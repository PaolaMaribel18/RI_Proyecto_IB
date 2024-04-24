@echo off

rem Ruta del directorio donde se encuentran los archivos de texto
set "directorio=D:\SEPTIMO SEMESTRE II\RECUPERACION INFORMACION\KevinMaldonado99\TASK 1\Books"

rem Solicitar al usuario que ingrese la palabra a buscar
set /p "palabra_buscada=Ingrese la palabra que desea buscar: "

echo La palabra '%palabra_buscada%' aparece en los siguientes archivos:
echo -----------------------------------------------------------

rem Buscar la palabra en todos los archivos de texto en el directorio
for %%F in ("%directorio%\*.txt") do (
    find /i "%palabra_buscada%" "%%F" >nul && (
        echo - Archivo: %%~nxF
    )
)

pause
