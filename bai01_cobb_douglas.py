    # =====================================================
    # 1.5
    # =====================================================
    with tabs[4]:
        st.header("1.5. Thảo luận chính sách")

        st.markdown("### a) TFP của Việt Nam có xu hướng tăng hay giảm?")

        st.success(
            f"TFP giai đoạn 2020-2025 có xu hướng **{trend}**, "
            f"thay đổi khoảng **{tfp_change:.2f}%**."
        )

        st.markdown(
            f"""
            Kết quả này cho thấy chất lượng tăng trưởng của Việt Nam trong giai đoạn 2020-2025
            có sự thay đổi đáng chú ý. TFP phản ánh phần tăng trưởng không được giải thích trực tiếp
            bởi các yếu tố đầu vào quan sát được như vốn vật chất, lao động, số hóa, AI và vốn nhân lực số.

            Nếu TFP tăng, điều đó hàm ý nền kinh tế đang sử dụng nguồn lực hiệu quả hơn.
            Nói cách khác, cùng một lượng vốn, lao động và công nghệ, nền kinh tế tạo ra nhiều sản lượng hơn.
            Đây là dấu hiệu tích cực vì tăng trưởng không chỉ đến từ mở rộng quy mô đầu vào,
            mà còn đến từ cải thiện công nghệ, quản trị, tổ chức sản xuất, chuyển đổi số và năng lực đổi mới sáng tạo.

            Ngược lại, nếu TFP giảm hoặc tăng rất chậm, điều này cho thấy tăng trưởng vẫn có thể phụ thuộc nhiều
            vào tích lũy vốn và mở rộng lao động. Khi đó, chất lượng tăng trưởng chưa thật sự bền vững,
            vì các nguồn lực truyền thống như vốn và lao động đều có giới hạn.
            """
        )

        st.markdown("### b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất?")

        st.success(
            f"Trong ba yếu tố mới **D, AI và H**, yếu tố đóng góp nổi bật nhất theo mô hình là "
            f"**{largest_new['Yếu tố']}**, chiếm khoảng "
            f"**{largest_new['Tỷ trọng tăng trưởng (%)']:.1f}%** trong tăng trưởng GDP bình quân."
        )

        st.markdown(
            f"""
            Trong mô hình Cobb-Douglas mở rộng, mức đóng góp của một yếu tố phụ thuộc vào hai thành phần chính:
            **tốc độ tăng của yếu tố đó** và **hệ số co giãn của yếu tố đó trong hàm sản xuất**.
            Vì vậy, một biến có tốc độ tăng nhanh nhưng hệ số nhỏ có thể đóng góp vừa phải;
            ngược lại, một biến tăng không quá nhanh nhưng có hệ số lớn vẫn có thể tạo tác động đáng kể.

            Về ý nghĩa kinh tế, **D** đại diện cho mức độ số hóa của nền kinh tế, thường được đo bằng tỷ trọng
            kinh tế số trong GDP. Khi D tăng, các hoạt động kinh tế được số hóa nhiều hơn, chi phí giao dịch giảm,
            khả năng kết nối thị trường tốt hơn và năng suất tổng thể có thể được cải thiện.

            **AI** đại diện cho năng lực trí tuệ nhân tạo hoặc số lượng doanh nghiệp công nghệ số.
            Sự gia tăng của AI cho thấy khu vực doanh nghiệp có khả năng ứng dụng công nghệ mới tốt hơn,
            từ đó hỗ trợ tự động hóa, phân tích dữ liệu, tối ưu sản xuất và nâng cao chất lượng dịch vụ.

            **H** đại diện cho vốn nhân lực số, tức tỷ lệ lao động qua đào tạo hoặc có kỹ năng phù hợp.
            Đây là điều kiện nền tảng để công nghệ phát huy hiệu quả. Nếu thiếu nhân lực số,
            đầu tư vào số hóa và AI có thể không chuyển hóa đầy đủ thành tăng trưởng thực tế.

            Do đó, dù yếu tố **{largest_new['Yếu tố']}** là yếu tố nổi bật nhất trong kết quả hiện tại,
            ba yếu tố D, AI và H không nên được xem là tách rời. Chúng có quan hệ bổ trợ lẫn nhau:
            kinh tế số cần doanh nghiệp công nghệ, doanh nghiệp công nghệ cần nhân lực số,
            và nhân lực số chỉ phát huy hiệu quả khi có môi trường số đủ phát triển.
            """
        )

        st.markdown("### c) Mục tiêu kinh tế số đạt 30% GDP vào năm 2030 có khả thi không?")

        if target_D >= 30:
            st.success(
                f"Trong kịch bản hiện tại, D năm 2030 đạt **{target_D:.1f}%**, "
                "tức đã phản ánh mục tiêu kinh tế số chiếm khoảng 30% GDP."
            )
        else:
            st.warning(
                f"Trong kịch bản hiện tại, D năm 2030 mới đạt **{target_D:.1f}%**, "
                "thấp hơn mục tiêu 30% GDP."
            )

        st.markdown(
            f"""
            Dựa trên mô hình mô phỏng, mục tiêu kinh tế số đạt khoảng **30% GDP vào năm 2030**
            là có cơ sở khả thi nếu các giả định chính được duy trì. Cụ thể, nền kinh tế cần đồng thời
            mở rộng tỷ trọng kinh tế số, gia tăng số lượng và năng lực doanh nghiệp công nghệ số,
            cải thiện chất lượng nhân lực và duy trì tăng trưởng TFP.

            Tuy nhiên, mục tiêu này không nên được hiểu đơn giản là chỉ cần tăng tỷ trọng D.
            Nếu D tăng nhưng AI và H không tăng tương ứng, tác động đến GDP có thể bị giới hạn.
            Nói cách khác, kinh tế số chỉ tạo ra tăng trưởng bền vững khi đi kèm với năng lực hấp thụ công nghệ
            của doanh nghiệp và kỹ năng số của người lao động.

            Để mục tiêu 30% kinh tế số/GDP có tính khả thi cao hơn, cần một số ràng buộc chính sách:

            - **Thứ nhất, hạ tầng số phải được đầu tư đồng bộ.**
            Điều này bao gồm băng rộng, trung tâm dữ liệu, điện toán đám mây, nền tảng số,
            định danh số, thanh toán số và an toàn thông tin.

            - **Thứ hai, doanh nghiệp phải chuyển đổi số thực chất.**
            Không chỉ tăng số lượng doanh nghiệp công nghệ, mà cần tăng mức độ ứng dụng công nghệ
            trong sản xuất, logistics, tài chính, thương mại, nông nghiệp, công nghiệp và dịch vụ công.

            - **Thứ ba, vốn nhân lực số phải được nâng cao.**
            Người lao động cần kỹ năng dữ liệu, kỹ năng sử dụng AI, kỹ năng vận hành nền tảng số
            và năng lực thích ứng với công nghệ mới.

            - **Thứ tư, TFP phải tiếp tục tăng.**
            Nếu TFP không cải thiện, nền kinh tế dễ quay lại mô hình tăng trưởng dựa vào vốn và lao động,
            trong khi hiệu quả dài hạn không cao.

            - **Thứ năm, thể chế dữ liệu và đổi mới sáng tạo cần hoàn thiện.**
            Cần có chính sách về chia sẻ dữ liệu, bảo mật, quyền riêng tư, cạnh tranh nền tảng,
            hỗ trợ doanh nghiệp nhỏ và vừa chuyển đổi số, cũng như khuyến khích nghiên cứu phát triển.

            **Kết luận:** mục tiêu kinh tế số đạt 30% GDP vào năm 2030 là có thể đạt được trong mô hình,
            nhưng chỉ khả thi nếu Việt Nam không chỉ mở rộng quy mô kinh tế số, mà còn nâng đồng thời
            **năng lực AI, vốn nhân lực số và năng suất nhân tố tổng hợp TFP**.
            """
        )

        st.markdown("### Kết luận chung")

        st.info(
            f"Mô hình Cobb-Douglas mở rộng cho thấy tăng trưởng GDP Việt Nam có thể được phân tích "
            f"thông qua các yếu tố K, L, D, AI, H và TFP. "
            f"TFP có xu hướng **{trend}**, MAPE của mô hình dự báo là **{mape:.2f}%**, "
            f"và trong nhóm yếu tố mới, yếu tố nổi bật nhất là **{largest_new['Yếu tố']}**. "
            f"Hàm ý chính sách là Việt Nam cần kết hợp chuyển đổi số, phát triển AI, nâng cao nhân lực số "
            f"và cải thiện TFP để đạt tăng trưởng bền vững."
        )
